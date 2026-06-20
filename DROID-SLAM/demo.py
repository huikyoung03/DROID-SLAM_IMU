import sys
sys.path.append("droid_slam")

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import argparse
import csv
import math

from droid import Droid
from droid_async import DroidAsync


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow("image", image / 255.0)
    cv2.waitKey(1)


def rotvec_to_quat_wxyz(rx, ry, rz):
    """
    rotation vector [rx, ry, rz] rad -> quaternion [w, x, y, z]
    """
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)

    if theta < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]

    ax = rx / theta
    ay = ry / theta
    az = rz / theta

    half = 0.5 * theta
    s = math.sin(half)

    return [
        math.cos(half),
        ax * s,
        ay * s,
        az * s,
    ]


def compute_imu_valid_and_weight(dt, imu_count, dr_norm_deg):
    """
    현재 imu_prior.csv에는 imu_valid / imu_weight가 없으므로
    demo.py에서 직접 계산한다.
    """

    if not math.isfinite(dt) or dt <= 0.0:
        return False, 0.0, "bad_dt"

    if dt > 0.5:
        return False, 0.0, "dt_too_large"

    if imu_count < 2:
        return False, 0.0, "too_few_imu"

    if not math.isfinite(dr_norm_deg):
        return False, 0.0, "bad_rotation"

    if dr_norm_deg < 0.01:
        return False, 0.0, "too_small_rotation"

    if dr_norm_deg > 90.0:
        return False, 0.0, "too_large_rotation"

    # 현재 적용 여부 확인이 목적이므로 기존 0.001보다 조금 크게 둠.
    # 실제 안정화 실험에서는 0.001~0.01 사이로 조절.
    return True, 0.01, "ok"


def load_imu_prior_csv(path):
    """
    imu_prior.csv를 frame_id 기준 dict로 로드한다.

    지원 형식 1:
        dq_w, dq_x, dq_y, dq_z, dr_norm_deg, imu_valid, imu_weight 포함

    지원 형식 2:
        현재 파일처럼 dr_x, dr_y, dr_z, dt, imu_count만 포함
        -> demo.py에서 quaternion / valid / weight 계산
    """

    priors = {}

    if path is None:
        return priors

    if not os.path.exists(path):
        print(f"[IMU] imu_prior path does not exist: {path}")
        return priors

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                frame_id = int(float(row.get("frame_id", -1)))

                dt = float(row.get("dt", 0.0))
                imu_count = int(float(row.get("imu_count", 0)))

                # -------------------------------------------------
                # Case A. 새 형식: dq, dr_norm_deg, valid, weight 있음
                # -------------------------------------------------
                if (
                    "dq_w" in row
                    and "dq_x" in row
                    and "dq_y" in row
                    and "dq_z" in row
                ):
                    dq_w = float(row.get("dq_w", 1.0))
                    dq_x = float(row.get("dq_x", 0.0))
                    dq_y = float(row.get("dq_y", 0.0))
                    dq_z = float(row.get("dq_z", 0.0))

                    dr_norm_deg = float(row.get("dr_norm_deg", 0.0))

                    raw_valid = row.get("imu_valid", "0")
                    if isinstance(raw_valid, str):
                        imu_valid = raw_valid.strip().lower() in ["1", "true", "yes", "y"]
                    else:
                        imu_valid = bool(raw_valid)

                    imu_weight = float(row.get("imu_weight", 0.0))
                    invalid_reason = str(row.get("invalid_reason", ""))

                # -------------------------------------------------
                # Case B. 현재 형식: dr_x, dr_y, dr_z만 있음
                # -------------------------------------------------
                else:
                    dr_x = float(row.get("dr_x", 0.0))
                    dr_y = float(row.get("dr_y", 0.0))
                    dr_z = float(row.get("dr_z", 0.0))

                    dr_norm = math.sqrt(dr_x * dr_x + dr_y * dr_y + dr_z * dr_z)
                    dr_norm_deg = dr_norm * 180.0 / math.pi

                    dq_w, dq_x, dq_y, dq_z = rotvec_to_quat_wxyz(
                        dr_x,
                        dr_y,
                        dr_z,
                    )

                    imu_valid, imu_weight, invalid_reason = compute_imu_valid_and_weight(
                        dt=dt,
                        imu_count=imu_count,
                        dr_norm_deg=dr_norm_deg,
                    )

                values = [
                    dq_w,
                    dq_x,
                    dq_y,
                    dq_z,
                    dt,
                    dr_norm_deg,
                    imu_weight,
                ]

                if any(not math.isfinite(v) for v in values):
                    imu_valid = False
                    imu_weight = 0.0
                    invalid_reason = "nan_or_inf_in_demo_loader"

                priors[frame_id] = {
                    "frame_id": frame_id,
                    "dt": dt,
                    "imu_count": imu_count,
                    "dr_norm_deg": dr_norm_deg,
                    "dq": [dq_w, dq_x, dq_y, dq_z],
                    "imu_valid": imu_valid,
                    "imu_weight": imu_weight,
                    "invalid_reason": invalid_reason,
                }

            except Exception as e:
                print(f"[IMU] failed to parse row: {row}, error={e}")

    valid_count = sum(1 for v in priors.values() if v["imu_valid"])
    weight_count = sum(1 for v in priors.values() if v["imu_weight"] > 0)

    print(f"[IMU] parsed priors: {len(priors)}")
    print(f"[IMU] valid priors: {valid_count}")
    print(f"[IMU] weight > 0: {weight_count}")

    return priors


def image_stream(imagedir, calib, stride):
    """image generator"""

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for idx, imfile in enumerate(image_list):
        image_path = os.path.join(imagedir, imfile)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[WARN] failed to read image: {image_path}")
            continue

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape

        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1 - h1 % 8, :w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0

        try:
            frame_id = int(os.path.splitext(os.path.basename(imfile))[0])
        except Exception:
            frame_id = idx

        yield frame_id, image[None], intrinsics


def save_reconstruction(droid, save_path):
    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value

    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu(),
    }

    torch.save(save_data, save_path)
    print(f"[SAVE] reconstruction saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")
    
    parser.add_argument("--beta",type=float,default=0.3,help="weight for translation / rotation components of flow",)
    parser.add_argument("--filter_thresh",type=float,default=2.4,help="how much motion before considering new keyframe",)
    parser.add_argument("--warmup",type=int,default=8,help="number of warmup frames",)
    parser.add_argument("--keyframe_thresh",type=float,default=4.0,help="threshold to create a new keyframe",)
    
    parser.add_argument("--frontend_thresh",type=float,default=16.0,help="add edges between frames within this distance",)
    parser.add_argument("--frontend_window",type=int,default=25,help="frontend optimization window",)
    parser.add_argument("--frontend_radius",type=int,default=2,help="force edges between frames within radius",)
    parser.add_argument("--frontend_nms",type=int,default=1,help="non-maximal suppression of edges",)
    
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
   
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")
    parser.add_argument("--reconstruction_path",help="path to saved reconstruction",)
    
    parser.add_argument("--imu_prior",type=str,default=None,help="path to imu_prior.csv",)
    parser.add_argument("--imu_mode",type=str,default="init_rotation",choices=["off", "init_rotation"],help="off: ignore IMU, init_rotation: use IMU as weak rotation initialization",)
    parser.add_argument("--imu_rotation_weight",type=float,default=1.0,help=("gain multiplied to imu_weight from csv. ""If imu_weight in csv is 0.001 and this is 1.0, final weight is 0.001."),)
    parser.add_argument("--imu_rotation_weight_max",type=float,default=0.01,help="safety cap for final IMU rotation weight",)
    parser.add_argument("--imu_compose_order",type=str,default="prev_dq", choices=["prev_dq", "dq_prev"],help=("quaternion composition order. ""Use prev_dq first. If direction looks wrong, try dq_prev."),)
    parser.add_argument("--imu_debug",action="store_true",help="print IMU application logs",)
    parser.add_argument("--imu_post_correction",action="store_true",help="apply IMU rotation-only post correction after DROID optimization",)
    parser.add_argument(
        "--imu_adaptive_init",
        action="store_true",
        help="use adaptive IMU rotation weight for init_rotation",
    )

    parser.add_argument(
        "--imu_adaptive_max_weight",
        type=float,
        default=0.003,
        help="maximum adaptive IMU init rotation weight",
    )

    args = parser.parse_args()

    args.stereo = False

    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    if args.reconstruction_path is not None:
        args.upsample = True

    imu_priors = None

    if args.imu_prior is not None:
        imu_priors = load_imu_prior_csv(args.imu_prior)

        print(f"[IMU] loaded imu priors: {len(imu_priors)} rows from {args.imu_prior}")

        if len(imu_priors) > 0:
            first_key = sorted(imu_priors.keys())[0]
            print(f"[IMU] first prior frame_id={first_key}: {imu_priors[first_key]}")

        if args.imu_mode == "off":
            print("[IMU] imu_mode=off, loaded priors will not be applied.")
        else:
            print(
                "[IMU] imu_mode=init_rotation, "
                f"rotation_weight_gain={args.imu_rotation_weight}, "
                f"rotation_weight_max={args.imu_rotation_weight_max}, "
                f"compose_order={args.imu_compose_order}"
            )

    droid = None

    for t, image, intrinsics in tqdm(
        image_stream(args.imagedir, args.calib, args.stride)
    ):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = DroidAsync(args) if args.asynchronous else Droid(args)

        imu_prior = None

        if imu_priors is not None and args.imu_mode != "off":
            imu_prior = imu_priors.get(int(t), None)

        droid.track(
            t,
            image,
            intrinsics=intrinsics,
            imu_prior=imu_prior,
        )

    if droid is None:
        raise RuntimeError("No images were processed. Check --imagedir and --stride.")

    traj_est = droid.terminate(
        image_stream(args.imagedir, args.calib, args.stride)
    )

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)