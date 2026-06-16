import os
import csv
import torch

from modules.corr import CorrBlock
from geom.projective_ops import coords_grid


MEAN = torch.as_tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
STD = torch.as_tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]


def quat_normalize(q):
    return q / torch.clamp(torch.linalg.norm(q), min=1e-8)


def quat_mul(q1, q2):
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]

    q = torch.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])

    return quat_normalize(q)


def rotvec_to_quat(rotvec):
    theta = torch.linalg.norm(rotvec)

    if float(theta.item()) < 1e-8:
        return torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=rotvec.device,
            dtype=torch.float32
        )

    axis = rotvec / theta
    half = theta * 0.5

    s = torch.sin(half)
    c = torch.cos(half)

    q = torch.cat([axis * s, c.view(1)])
    return quat_normalize(q)


class MotionFilter:
    """
    DROID-SLAM motion filter + aligned relative IMU rotation prior.

    핵심:
    - raw IMU가 아니라 imu_prior_aligned.csv 사용
    - IMU를 absolute pose로 직접 넣지 않음
    - 이전 accepted keyframe 이후의 aligned IMU rotation만 누적
    - 새 keyframe append 시 previous pose rotation에 relative IMU rotation을 compose
    """

    def __init__(self, net, video, thresh=2.5, imu_rotation_weight=1.0):
        self.model = net
        self.video = video
        self.thresh = float(thresh)
        self.count = 0

        self.net = None
        self.inp = None
        self.fmap = None

        self.imu_rotation_weight = float(imu_rotation_weight)

        # 이전 keyframe 이후 누적 relative rotation
        self.pending_rotvec = torch.zeros(3, device="cuda", dtype=torch.float32)

        # 한 keyframe 구간에서 IMU prior가 너무 크게 들어가지 않도록 제한
        # 기본 5도
        self.max_pending_angle_rad = float(
            os.environ.get("DROID_IMU_MAX_PENDING_RAD", "0.0872665")
        )

        # q_prior = q_prev ⊗ dq 또는 dq ⊗ q_prev 실험 가능
        self.compose_order = os.environ.get("DROID_IMU_COMPOSE_ORDER", "right")

        self.debug_imu_log_path = os.environ.get("DROID_IMU_DEBUG_LOG", None)
        self.debug_imu_log_initialized = False

        ht = video.ht // 8
        wd = video.wd // 8
        self.coords0 = coords_grid(ht, wd, device="cuda")[None, None]

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        return self.model.fnet(image)

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        net, inp = self.model.cnet(image).split([128, 128], dim=2)
        return net.tanh(), inp.relu()

    def _identity_pose(self):
        return torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            device="cuda",
            dtype=torch.float32
        )

    def _get_previous_pose(self):
        if self.video.counter.value <= 0:
            return self._identity_pose()

        prev_index = self.video.counter.value - 1
        pose = self.video.poses[prev_index].clone().detach().float()

        if not torch.isfinite(pose).all():
            return self._identity_pose()

        pose[3:] = quat_normalize(pose[3:])
        return pose

    def _update_pending_imu(self, imu_prior):
        if imu_prior is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            dt = float(imu_prior.get("dt", 0.0))
            imu_count = float(imu_prior.get("imu_count", 0.0))
            dr_x = float(imu_prior.get("dr_x", 0.0))
            dr_y = float(imu_prior.get("dr_y", 0.0))
            dr_z = float(imu_prior.get("dr_z", 0.0))

            if dt <= 0 or imu_count <= 0:
                return 0.0, dt, imu_count, dr_x, dr_y, dr_z

            dr = torch.tensor(
                [dr_x, dr_y, dr_z],
                device="cuda",
                dtype=torch.float32
            )

            dr = dr * self.imu_rotation_weight

            if torch.isfinite(dr).all():
                self.pending_rotvec = self.pending_rotvec + dr

            return float(torch.linalg.norm(dr).item()), dt, imu_count, dr_x, dr_y, dr_z

        except Exception:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def _make_relative_imu_pose_prior(self):
        prev_pose = self._get_previous_pose()
        pose_prior = prev_pose.clone()

        pending = self.pending_rotvec.clone()
        pending_norm = torch.linalg.norm(pending)

        if float(pending_norm.item()) > self.max_pending_angle_rad:
            pending = pending / pending_norm * self.max_pending_angle_rad
            pending_norm = torch.linalg.norm(pending)

        dq = rotvec_to_quat(pending)
        q_prev = quat_normalize(prev_pose[3:])

        if self.compose_order == "left":
            q_prior = quat_mul(dq, q_prev)
        else:
            q_prior = quat_mul(q_prev, dq)

        pose_prior[3:] = quat_normalize(q_prior)

        if not torch.isfinite(pose_prior).all():
            return self._identity_pose(), float(pending_norm.item()), 1

        return pose_prior, float(pending_norm.item()), 0

    def _write_imu_debug_log(
        self,
        tstamp,
        motn,
        appended,
        dr_norm,
        dt,
        imu_count,
        dr_x,
        dr_y,
        dr_z,
        pending_norm,
        prior_has_nan,
        pose_prior
    ):
        if self.debug_imu_log_path is None:
            return

        header = [
            "frame_id",
            "motn",
            "appended",
            "dt",
            "imu_count",
            "dr_x",
            "dr_y",
            "dr_z",
            "dr_norm",
            "dr_norm_deg",
            "pending_norm",
            "pending_norm_deg",
            "imu_weight",
            "compose_order",
            "prior_tx",
            "prior_ty",
            "prior_tz",
            "prior_qx",
            "prior_qy",
            "prior_qz",
            "prior_qw",
            "prior_q_norm",
            "prior_has_nan",
        ]

        write_header = False
        if not self.debug_imu_log_initialized:
            if not os.path.exists(self.debug_imu_log_path):
                write_header = True
            self.debug_imu_log_initialized = True

        pp = pose_prior.detach().float().cpu()
        qnorm = float(torch.linalg.norm(pp[3:]).item())

        with open(self.debug_imu_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)

            writer.writerow([
                int(tstamp),
                float(motn),
                int(bool(appended)),
                float(dt),
                float(imu_count),
                float(dr_x),
                float(dr_y),
                float(dr_z),
                float(dr_norm),
                float(dr_norm) * 57.29577951308232,
                float(pending_norm),
                float(pending_norm) * 57.29577951308232,
                float(self.imu_rotation_weight),
                self.compose_order,
                float(pp[0]),
                float(pp[1]),
                float(pp[2]),
                float(pp[3]),
                float(pp[4]),
                float(pp[5]),
                float(pp[6]),
                qnorm,
                int(prior_has_nan),
            ])

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None, imu_prior=None):
        inputs = image[None, :, [2, 1, 0]].cuda() / 255.0
        inputs = inputs.sub_(MEAN).div_(STD)

        gmap = self.__feature_encoder(inputs)

        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:, [0]])

            self.net = net
            self.inp = inp
            self.fmap = gmap
            self.pending_rotvec.zero_()

            self.video.append(
                tstamp,
                image[0],
                self._identity_pose(),
                1.0,
                depth,
                intrinsics / 8.0,
                gmap,
                net[0, 0],
                inp[0, 0]
            )
            return

        # 매 입력 프레임마다 aligned IMU rotation 누적
        dr_norm, dt, imu_count, dr_x, dr_y, dr_z = self._update_pending_imu(imu_prior)

        corr = CorrBlock(self.fmap, gmap)(self.coords0)

        self.net, delta, weight = self.model.update(
            self.net,
            self.inp,
            corr
        )

        motn = torch.linalg.norm(delta[0, 0].mean(dim=[0, 1])).item()
        appended = motn > self.thresh

        pose_prior = self._get_previous_pose()
        pending_norm = float(torch.linalg.norm(self.pending_rotvec).item())
        prior_has_nan = 0

        if appended:
            self.count = 0

            net, inp = self.__context_encoder(inputs[:, [0]])

            self.net = net
            self.inp = inp
            self.fmap = gmap

            if self.imu_rotation_weight > 0.0:
                pose_prior, pending_norm, prior_has_nan = self._make_relative_imu_pose_prior()
            else:
                pose_prior = self._identity_pose()
                pending_norm = 0.0
                prior_has_nan = 0

            self.video.append(
                tstamp,
                image[0],
                pose_prior,
                None,
                depth,
                intrinsics / 8.0,
                gmap,
                net[0, 0],
                inp[0, 0]
            )

            # keyframe accepted 이후 pending IMU reset
            self.pending_rotvec.zero_()

        else:
            self.count += 1

        self._write_imu_debug_log(
            tstamp=tstamp,
            motn=motn,
            appended=appended,
            dr_norm=dr_norm,
            dt=dt,
            imu_count=imu_count,
            dr_x=dr_x,
            dr_y=dr_y,
            dr_z=dr_z,
            pending_norm=pending_norm,
            prior_has_nan=prior_has_nan,
            pose_prior=pose_prior
        )

        if imu_prior is not None and int(tstamp) % 20 == 0:
            print(
                f"[IMU-REL] frame={int(tstamp)} "
                f"motn={motn:.4f} "
                f"append={appended} "
                f"dr_deg={dr_norm * 57.29577951308232:.3f} "
                f"pending_deg={pending_norm * 57.29577951308232:.3f} "
                f"order={self.compose_order}"
            )
