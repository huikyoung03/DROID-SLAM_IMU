import csv
import math
import torch


def load_imu_prior_csv(path):
    """
    imu_prior.csv를 frame_id 기준 dictionary로 로드한다.

    expected columns:
    frame_id, timestamp_sec, dt,
    dr_x, dr_y, dr_z,
    dv_x, dv_y, dv_z,
    dp_x, dp_y, dp_z,
    imu_count
    """

    imu_priors = {}

    if path is None:
        return None

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if "frame_id" not in row:
                continue

            frame_id = int(float(row["frame_id"]))

            def get_float(key, default=0.0):
                value = row.get(key, default)
                if value is None or value == "":
                    return default
                try:
                    return float(value)
                except Exception:
                    return default

            imu_priors[frame_id] = {
                "frame_id": frame_id,
                "timestamp_sec": get_float("timestamp_sec", 0.0),
                "prev_timestamp_sec": get_float("prev_timestamp_sec", 0.0),
                "dt": get_float("dt", 0.0),

                "dr_x": get_float("dr_x", 0.0),
                "dr_y": get_float("dr_y", 0.0),
                "dr_z": get_float("dr_z", 0.0),

                "dv_x": get_float("dv_x", 0.0),
                "dv_y": get_float("dv_y", 0.0),
                "dv_z": get_float("dv_z", 0.0),

                "dp_x": get_float("dp_x", 0.0),
                "dp_y": get_float("dp_y", 0.0),
                "dp_z": get_float("dp_z", 0.0),

                "imu_count": get_float("imu_count", 0.0),
            }

    return imu_priors


def rotvec_to_quat_torch(rotvec, device="cuda", dtype=torch.float32):
    """
    rotation vector [rx, ry, rz]를 quaternion [qx, qy, qz, qw]로 변환한다.
    rotvec 단위는 rad 기준이다.
    """

    if not torch.is_tensor(rotvec):
        rotvec = torch.tensor(rotvec, device=device, dtype=dtype)
    else:
        rotvec = rotvec.to(device=device, dtype=dtype)

    angle = torch.linalg.norm(rotvec)

    eps = torch.tensor(1e-8, device=device, dtype=dtype)

    if angle.item() < 1e-8:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)

    axis = rotvec / torch.maximum(angle, eps)
    half = 0.5 * angle

    sin_half = torch.sin(half)
    cos_half = torch.cos(half)

    qx = axis[0] * sin_half
    qy = axis[1] * sin_half
    qz = axis[2] * sin_half
    qw = cos_half

    quat = torch.stack([qx, qy, qz, qw])
    quat = quat / torch.linalg.norm(quat).clamp(min=1e-8)

    return quat


def quat_multiply_torch(q1, q2):
    """
    quaternion multiply.
    q format: [qx, qy, qz, qw]
    return: q1 * q2
    """

    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    q = torch.stack([x, y, z, w])
    q = q / torch.linalg.norm(q).clamp(min=1e-8)

    return q


def imu_prior_to_delta_quat(imu_prior, rotation_weight=1.0, device="cuda"):
    """
    imu_prior row에서 dr_x, dr_y, dr_z를 읽어 delta quaternion으로 변환한다.
    rotation_weight는 IMU 회전 prior 강도 조절용이다.
    """

    if imu_prior is None:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)

    dr = torch.tensor([
        float(imu_prior.get("dr_x", 0.0)),
        float(imu_prior.get("dr_y", 0.0)),
        float(imu_prior.get("dr_z", 0.0)),
    ], device=device, dtype=torch.float32)

    dr = dr * float(rotation_weight)

    return rotvec_to_quat_torch(dr, device=device, dtype=torch.float32)


class ImuRotationAccumulator:
    """
    프레임별 IMU 회전량을 누적해서 absolute-like rotation prior를 만든다.

    DROID-SLAM pose는 SE3 형태 [tx, ty, tz, qx, qy, qz, qw]로 들어간다.
    여기서는 translation은 직접 신뢰하지 않고, rotation만 누적한다.
    """

    def __init__(self, rotation_weight=1.0, device="cuda"):
        self.rotation_weight = rotation_weight
        self.device = device
        self.reset()

    def reset(self):
        self.q = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=self.device,
            dtype=torch.float32
        )

    def update(self, imu_prior):
        if imu_prior is None:
            return self.q

        imu_count = float(imu_prior.get("imu_count", 0.0))
        dt = float(imu_prior.get("dt", 0.0))

        # IMU가 없거나 dt가 비정상인 구간은 누적하지 않는다.
        if imu_count <= 0 or dt <= 0:
            return self.q

        dq = imu_prior_to_delta_quat(
            imu_prior,
            rotation_weight=self.rotation_weight,
            device=self.device
        )

        self.q = quat_multiply_torch(self.q, dq)
        self.q = self.q / torch.linalg.norm(self.q).clamp(min=1e-8)

        return self.q

    def get_se3_vec(self):
        """
        lietorch.SE3 입력용 7-vector 반환.
        [tx, ty, tz, qx, qy, qz, qw]
        translation은 0으로 둔다.
        """

        t = torch.zeros(3, device=self.device, dtype=torch.float32)
        return torch.cat([t, self.q], dim=0)