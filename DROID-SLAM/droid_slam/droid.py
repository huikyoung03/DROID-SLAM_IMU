import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


def _quat_normalize_xyzw(q):
    """
    Quaternion normalize.
    DROID pose quaternion order is assumed as [x, y, z, w].
    """
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)


def _quat_mul_xyzw(q1, q2):
    """
    Quaternion multiplication.

    q1, q2 order:
        [x, y, z, w]

    return:
        q1 * q2
    """
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    out = torch.stack([x, y, z, w], dim=-1)
    return _quat_normalize_xyzw(out)


def _slerp_identity_xyzw(q, weight):
    """
    Apply only a small fraction of IMU rotation.

    q:
        target delta quaternion [x, y, z, w]

    weight:
        0.0 -> identity rotation
        1.0 -> full IMU rotation

    For safety, this project should usually use a very small weight:
        0.0001 ~ 0.01
    """
    q = _quat_normalize_xyzw(q)

    # q and -q represent the same rotation.
    # Make w positive to choose the shorter rotation.
    if q[..., 3] < 0:
        q = -q

    weight = float(weight)

    identity = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        device=q.device,
        dtype=q.dtype,
    )

    if weight <= 0.0:
        return identity

    if weight >= 1.0:
        return q

    w = torch.clamp(q[3], -1.0, 1.0)
    angle = 2.0 * torch.acos(w)

    if torch.abs(angle) < 1e-8:
        return identity

    sin_half = torch.sin(angle / 2.0)

    if torch.abs(sin_half) < 1e-8:
        return identity

    axis = q[0:3] / sin_half
    new_angle = angle * weight

    xyz = axis * torch.sin(new_angle / 2.0)
    qw = torch.cos(new_angle / 2.0).view(1)

    out = torch.cat([xyz, qw], dim=0)
    return _quat_normalize_xyzw(out)


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()

        self.args = args
        self.disable_vis = args.disable_vis
        self.visualizer = None

        self.load_weights(args.weights)

        # store images, depth, poses, intrinsics
        self.video = DepthVideo(
            args.image_size,
            args.buffer,
            stereo=args.stereo,
        )

        # ---------------------------------------------------------
        # IMPORTANT
        # ---------------------------------------------------------
        # MotionFilter는 기존 DROID-SLAM 방식대로 visual motion만 보고
        # frame append 여부를 결정하게 둔다.
        #
        # IMU를 MotionFilter에 직접 넣으면 timestamp/좌표계/노이즈 문제로
        # keyframe 선택이 흔들려 결과가 더 망가질 수 있다.
        # ---------------------------------------------------------
        self.filterx = MotionFilter(
            self.net,
            self.video,
            thresh=args.filter_thresh,
        )

        # frontend process
        self.frontend = DroidFrontend(
            self.net,
            self.video,
            self.args,
        )

        # backend process
        self.backend = DroidBackend(
            self.net,
            self.video,
            self.args,
        )

        # visualizer
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn

            self.visualizer = Process(
                target=visualization_fn,
                args=(self.video, None),
            )
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(
            self.net,
            self.video,
        )

    def load_weights(self, weights):
        """load trained model weights"""

        print(weights)

        self.net = DroidNet()

        state_dict = OrderedDict([
            (k.replace("module.", ""), v)
            for (k, v) in torch.load(weights).items()
        ])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def _is_valid_imu_prior(self, imu_prior):
        """
        imu_prior.csv에서 읽어온 IMU prior가 사용 가능한지 검사한다.

        expected imu_prior example:
            {
                "frame_id": 12,
                "dt": 0.101,
                "imu_count": 20,
                "dr_norm_deg": 3.5,
                "dq": [w, x, y, z],
                "imu_valid": True,
                "imu_weight": 0.001,
                "invalid_reason": "ok"
            }
        """
        if imu_prior is None:
            return False

        if not bool(imu_prior.get("imu_valid", False)):
            return False

        imu_weight = float(imu_prior.get("imu_weight", 0.0))
        if imu_weight <= 0.0:
            return False

        dq = imu_prior.get("dq", None)
        if dq is None or len(dq) != 4:
            return False

        for value in dq:
            if not np.isfinite(float(value)):
                return False

        dt = float(imu_prior.get("dt", 0.0))
        imu_count = int(imu_prior.get("imu_count", 0))
        dr_norm_deg = float(imu_prior.get("dr_norm_deg", 0.0))

        if dt <= 0.0 or dt > 0.5:
            return False

        if imu_count < 2:
            return False

        if dr_norm_deg < 0.05:
            return False

        if dr_norm_deg > 45.0:
            return False

        return True

    def _apply_imu_rotation_init(self, ix, imu_prior):
        """
        새 frame이 append된 직후, frontend/backend 최적화가 돌기 전에
        IMU delta rotation을 현재 frame pose 초기값에 약하게 반영한다.

        주의:
        - translation은 절대 건드리지 않는다.
        - DROID pose를 IMU로 덮어쓰지 않는다.
        - 이전 pose rotation에 IMU delta rotation을 아주 작게 합성한다.
        """
        if ix <= 0:
            return

        if not self._is_valid_imu_prior(imu_prior):
            return

        imu_mode = getattr(self.args, "imu_mode", "init_rotation")
        if imu_mode == "off":
            return

        if imu_mode != "init_rotation":
            return

        device = self.video.poses.device
        dtype = self.video.poses.dtype

        raw_weight = float(imu_prior.get("imu_weight", 0.0))

        # imu_rotation_weight는 전체 gain 역할.
        # prior 자체의 weight가 0.001이면,
        # imu_rotation_weight=1.0일 때 0.001만 반영된다.
        gain = float(getattr(self.args, "imu_rotation_weight", 1.0))

        # 너무 크게 들어가는 것 방지
        max_weight = float(getattr(self.args, "imu_rotation_weight_max", 0.01))
        weight = max(0.0, min(raw_weight * gain, max_weight))

        if weight <= 0.0:
            return

        dq_wxyz = imu_prior.get("dq", [1.0, 0.0, 0.0, 0.0])

        dq_w = float(dq_wxyz[0])
        dq_x = float(dq_wxyz[1])
        dq_y = float(dq_wxyz[2])
        dq_z = float(dq_wxyz[3])

        # imu_preintegrate.py는 [w, x, y, z]로 저장.
        # DROID pose는 일반적으로 [tx, ty, tz, qx, qy, qz, qw]라서
        # quaternion만 [x, y, z, w]로 변환한다.
        dq_xyzw = torch.tensor(
            [dq_x, dq_y, dq_z, dq_w],
            device=device,
            dtype=dtype,
        )

        dq_xyzw = _quat_normalize_xyzw(dq_xyzw)

        # IMU 회전을 그대로 쓰지 말고 작은 비율만 적용
        dq_small = _slerp_identity_xyzw(dq_xyzw, weight)

        prev_pose = self.video.poses[ix - 1].clone()
        cur_pose = self.video.poses[ix].clone()

        prev_q = prev_pose[3:7].clone()
        prev_q = _quat_normalize_xyzw(prev_q)

        compose_order = getattr(self.args, "imu_compose_order", "prev_dq")

        if compose_order == "dq_prev":
            pred_q = _quat_mul_xyzw(dq_small, prev_q)
        else:
            pred_q = _quat_mul_xyzw(prev_q, dq_small)

        # translation은 기존 DROID 초기값 유지
        cur_pose[3:7] = pred_q

        self.video.poses[ix] = cur_pose

        if bool(getattr(self.args, "imu_debug", False)):
            print(
                "[IMU] init_rotation "
                f"frame={ix}, "
                f"raw_weight={raw_weight:.8f}, "
                f"gain={gain:.4f}, "
                f"used_weight={weight:.8f}, "
                f"dr_norm_deg={float(imu_prior.get('dr_norm_deg', 0.0)):.4f}, "
                f"imu_count={int(imu_prior.get('imu_count', 0))}, "
                f"dt={float(imu_prior.get('dt', 0.0)):.4f}"
            )

    def track(self, tstamp, image, depth=None, intrinsics=None, imu_prior=None):
        """
        main thread - update map

        변경점:
        - MotionFilter에는 IMU를 넘기지 않는다.
        - frame이 실제 append된 경우에만 IMU rotation prior를 적용한다.
        - 적용 위치는 frontend/backend 최적화 전이다.
        """
        with torch.no_grad():
            before_counter = self.video.counter.value

            # 원본 DROID-SLAM 흐름 유지
            self.filterx.track(
                tstamp,
                image,
                depth,
                intrinsics,
            )

            after_counter = self.video.counter.value

            # MotionFilter가 frame을 실제로 append한 경우에만 적용
            if after_counter > before_counter:
                ix = after_counter - 1
                self._apply_imu_rotation_init(ix, imu_prior)

            counter = self.video.counter.value
            warmup = getattr(self.args, "warmup", 8)

            # local bundle adjustment
            if counter == warmup:
                self.frontend()

            # global bundle adjustment
            elif counter > warmup:
                self.frontend()
                self.backend(2)

            if self.visualizer is not None:
                self.visualizer()

    def terminate(self, stream=None):
        """terminate the visualization process, return poses [t, q]"""

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()