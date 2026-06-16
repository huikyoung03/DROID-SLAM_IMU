import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def q_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def q_inv(q):
    q = q_normalize(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def q_mul(q1, q2):
    x1, y1, z1, w1 = q_normalize(q1)
    x2, y2, z2, w2 = q_normalize(q2)

    return q_normalize(np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64))


def q_to_rotvec(q):
    q = q_normalize(q)
    x, y, z, w = q

    if w < 0:
        x, y, z, w = -x, -y, -z, -w

    v = np.array([x, y, z], dtype=np.float64)
    s = np.linalg.norm(v)

    if s < 1e-12:
        return np.zeros(3, dtype=np.float64)

    angle = 2.0 * np.arctan2(s, w)

    if angle > np.pi:
        angle -= 2.0 * np.pi

    axis = v / s
    return axis * angle


def estimate_R_imu_to_cam(imu_vecs, cam_vecs):
    """
    imu_vecs: [N, 3]
    cam_vecs: [N, 3]

    Find R such that:
        cam_vec ≈ R @ imu_vec
    """
    A = np.asarray(imu_vecs, dtype=np.float64)
    B = np.asarray(cam_vecs, dtype=np.float64)

    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    pred = (R @ A.T).T
    err = np.linalg.norm(pred - B, axis=1)

    return R, err, S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", required=True)
    parser.add_argument("--imu_prior", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--min_angle_deg", type=float, default=0.3)
    parser.add_argument("--max_angle_deg", type=float, default=20.0)
    args = parser.parse_args()

    recon_path = Path(args.reconstruction)
    imu_path = Path(args.imu_prior)

    data = torch.load(recon_path, map_location="cpu")

    poses = data["poses"].float().numpy()
    tstamps = data["tstamps"]

    if hasattr(tstamps, "cpu"):
        tstamps = tstamps.cpu().numpy()
    else:
        tstamps = np.asarray(tstamps)

    df = pd.read_csv(imu_path)

    required = ["frame_id", "dt", "imu_count", "dr_x", "dr_y", "dr_z"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"missing column in imu_prior.csv: {c}")

    df_by_frame = df.set_index("frame_id", drop=False)

    pairs = []

    for i in range(1, len(tstamps)):
        f0 = int(round(float(tstamps[i - 1])))
        f1 = int(round(float(tstamps[i])))

        if f1 <= f0:
            continue

        q0 = poses[i - 1, 3:7]
        q1 = poses[i, 3:7]

        if not np.isfinite(q0).all() or not np.isfinite(q1).all():
            continue

        q0 = q_normalize(q0)
        q1 = q_normalize(q1)

        # visual relative rotation 후보 2개
        q_rel_a = q_mul(q1, q_inv(q0))
        q_rel_b = q_mul(q_inv(q0), q1)

        rv_cam_a = q_to_rotvec(q_rel_a)
        rv_cam_b = q_to_rotvec(q_rel_b)

        # f0 다음 프레임부터 f1까지 IMU rotation 누적
        rv_imu = np.zeros(3, dtype=np.float64)
        used_rows = 0

        for fid in range(f0 + 1, f1 + 1):
            if fid not in df_by_frame.index:
                continue

            row = df_by_frame.loc[fid]

            if float(row["dt"]) <= 0 or float(row["imu_count"]) <= 0:
                continue

            rv_imu += np.array([
                float(row["dr_x"]),
                float(row["dr_y"]),
                float(row["dr_z"]),
            ], dtype=np.float64)

            used_rows += 1

        if used_rows == 0:
            continue

        imu_deg = np.rad2deg(np.linalg.norm(rv_imu))
        cam_deg_a = np.rad2deg(np.linalg.norm(rv_cam_a))
        cam_deg_b = np.rad2deg(np.linalg.norm(rv_cam_b))

        if imu_deg < args.min_angle_deg or imu_deg > args.max_angle_deg:
            continue

        pairs.append({
            "from": f0,
            "to": f1,
            "rv_imu": rv_imu,
            "rv_cam_a": rv_cam_a,
            "rv_cam_b": rv_cam_b,
            "imu_deg": imu_deg,
            "cam_deg_a": cam_deg_a,
            "cam_deg_b": cam_deg_b,
            "used_rows": used_rows,
        })

    print("selected pairs:", len(pairs))

    if len(pairs) < 5:
        print("Not enough pairs.")
        print("Try: --min_angle_deg 0.1 --max_angle_deg 30")
        raise SystemExit(1)

    imu_vecs = np.stack([p["rv_imu"] for p in pairs], axis=0)

    candidates = []

    for cam_key in ["rv_cam_a", "rv_cam_b"]:
        cam_vecs_raw = np.stack([p[cam_key] for p in pairs], axis=0)

        for sign in [1.0, -1.0]:
            cam_vecs = cam_vecs_raw * sign

            R, err, S = estimate_R_imu_to_cam(imu_vecs, cam_vecs)
            pred = (R @ imu_vecs.T).T

            angle_err_deg = np.rad2deg(np.linalg.norm(pred - cam_vecs, axis=1))

            candidates.append({
                "cam_key": cam_key,
                "sign": sign,
                "R": R,
                "S": S,
                "median_error_deg": float(np.median(angle_err_deg)),
                "mean_error_deg": float(np.mean(angle_err_deg)),
                "max_error_deg": float(np.max(angle_err_deg)),
            })

    candidates = sorted(candidates, key=lambda x: x["median_error_deg"])
    best = candidates[0]
    R = best["R"]

    print("\n=== BEST ALIGNMENT ===")
    print("cam_key:", best["cam_key"])
    print("sign:", best["sign"])
    print("det(R):", np.linalg.det(R))
    print("median error deg:", best["median_error_deg"])
    print("mean error deg:", best["mean_error_deg"])
    print("max error deg:", best["max_error_deg"])
    print("R_imu_to_cam:")
    print(R)

    print("\n=== all candidates ===")
    for c in candidates:
        print(
            c["cam_key"],
            "sign", c["sign"],
            "median", c["median_error_deg"],
            "mean", c["mean_error_deg"],
            "max", c["max_error_deg"]
        )

    out = {
        "R_imu_to_cam": R.tolist(),
        "det": float(np.linalg.det(R)),
        "selected_pairs": int(len(pairs)),
        "cam_key": best["cam_key"],
        "sign": float(best["sign"]),
        "median_error_deg": best["median_error_deg"],
        "mean_error_deg": best["mean_error_deg"],
        "max_error_deg": best["max_error_deg"],
        "source_reconstruction": str(recon_path),
        "source_imu_prior": str(imu_path),
    }

    Path(args.out_json).write_text(json.dumps(out, indent=2))

    aligned = df.copy()
    sign = float(best["sign"])

    dr_aligned = []

    for _, row in aligned.iterrows():
        rv = np.array([
            float(row["dr_x"]),
            float(row["dr_y"]),
            float(row["dr_z"]),
        ], dtype=np.float64)

        rv_cam = sign * (R @ rv)
        dr_aligned.append(rv_cam)

    dr_aligned = np.asarray(dr_aligned)

    aligned["dr_raw_x"] = aligned["dr_x"]
    aligned["dr_raw_y"] = aligned["dr_y"]
    aligned["dr_raw_z"] = aligned["dr_z"]

    aligned["dr_x"] = dr_aligned[:, 0]
    aligned["dr_y"] = dr_aligned[:, 1]
    aligned["dr_z"] = dr_aligned[:, 2]

    aligned.to_csv(args.out_csv, index=False)

    print("\nsaved json:", args.out_json)
    print("saved csv:", args.out_csv)


if __name__ == "__main__":
    main()
