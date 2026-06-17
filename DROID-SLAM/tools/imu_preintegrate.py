from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def convert_gyro_device_to_camera(gx: float, gy: float, gz: float) -> Tuple[float, float, float]:
    """
    Android device gyro -> camera 좌표계 변환 위치.

    초기 버전은 원본 축 그대로 사용한다.
    baseline보다 결과가 무너지면 아래 후보처럼 축/부호를 바꿔 실험한다.
      - return gx, -gy, -gz
      - return -gy, gx, gz
      - return gy, -gx, gz
      - return gx, gz, -gy
    """
    return gx, gy, gz


def rotvec_to_quat_wxyz(rx: float, ry: float, rz: float) -> Tuple[float, float, float, float]:
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    ax = rx / theta
    ay = ry / theta
    az = rz / theta
    half = 0.5 * theta
    s = math.sin(half)
    return math.cos(half), ax * s, ay * s, az * s


def compute_imu_weight(dt: float, imu_count: int, dr_norm_deg: float, has_nan: bool) -> Tuple[float, int, str]:
    if has_nan:
        return 0.0, 0, "nan"
    if imu_count < 2:
        return 0.0, 0, "too_few_imu"
    if dt <= 0.0:
        return 0.0, 0, "bad_dt"
    if dt > 0.5:
        return 0.0, 0, "dt_too_large"
    if dr_norm_deg < 0.05:
        return 0.0, 0, "too_small_rotation"
    if dr_norm_deg > 45.0:
        return 0.0, 0, "too_large_rotation"
    if dr_norm_deg > 20.0:
        return 0.0003, 1, "large_but_accepted"
    return 0.001, 1, "ok"


def load_frames(frames_csv: Path) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    with open(frames_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frames.append({
                "frame_id": int(row["frame_id"]),
                "frame_index": len(frames),
                "timestamp_sec": float(row["timestamp_sec"]),
                "timestamp_ns": int(float(row["timestamp_ns"])),
                "filename": row.get("filename", f"{len(frames):06d}.jpg"),
            })
    frames.sort(key=lambda x: x["timestamp_ns"])
    return frames


def load_imu(imu_csv: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(imu_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gx, gy, gz = convert_gyro_device_to_camera(
                to_float(row.get("gx")),
                to_float(row.get("gy")),
                to_float(row.get("gz")),
            )
            samples.append({
                "timestamp_sec": to_float(row.get("timestamp_sec")),
                "timestamp_ns": int(float(row.get("timestamp_ns", 0))),
                "gx": gx,
                "gy": gy,
                "gz": gz,
                "ax": to_float(row.get("ax")),
                "ay": to_float(row.get("ay")),
                "az": to_float(row.get("az")),
            })
    samples.sort(key=lambda x: x["timestamp_ns"])
    return samples


def integrate_gyro_window(imu_window: List[Dict[str, Any]], start_ns: Optional[int], end_ns: int) -> Dict[str, Any]:
    if start_ns is None or len(imu_window) < 2:
        dt = 0.0 if start_ns is None else (end_ns - start_ns) / 1e9
        dq = rotvec_to_quat_wxyz(0.0, 0.0, 0.0)
        return {
            "dt": dt,
            "imu_count": len(imu_window),
            "dr_x": 0.0,
            "dr_y": 0.0,
            "dr_z": 0.0,
            "dr_norm": 0.0,
            "dr_norm_deg": 0.0,
            "dq_w": dq[0],
            "dq_x": dq[1],
            "dq_y": dq[2],
            "dq_z": dq[3],
            "has_nan": False,
        }

    dr_x = 0.0
    dr_y = 0.0
    dr_z = 0.0

    for prev, curr in zip(imu_window[:-1], imu_window[1:]):
        dt_sample = (curr["timestamp_ns"] - prev["timestamp_ns"]) / 1e9
        if dt_sample <= 0.0 or dt_sample > 0.2:
            continue
        gx = 0.5 * (prev["gx"] + curr["gx"])
        gy = 0.5 * (prev["gy"] + curr["gy"])
        gz = 0.5 * (prev["gz"] + curr["gz"])
        dr_x += gx * dt_sample
        dr_y += gy * dt_sample
        dr_z += gz * dt_sample

    dr_norm = math.sqrt(dr_x * dr_x + dr_y * dr_y + dr_z * dr_z)
    dr_norm_deg = math.degrees(dr_norm)
    dq = rotvec_to_quat_wxyz(dr_x, dr_y, dr_z)
    values = [dr_x, dr_y, dr_z, dr_norm, dr_norm_deg, *dq]
    has_nan = any(math.isnan(v) or math.isinf(v) for v in values)

    return {
        "dt": (end_ns - start_ns) / 1e9,
        "imu_count": len(imu_window),
        "dr_x": dr_x,
        "dr_y": dr_y,
        "dr_z": dr_z,
        "dr_norm": dr_norm,
        "dr_norm_deg": dr_norm_deg,
        "dq_w": dq[0],
        "dq_x": dq[1],
        "dq_y": dq[2],
        "dq_z": dq[3],
        "has_nan": has_nan,
    }


def build_imu_prior(frames_csv: Path, imu_csv: Path, output_csv: Path) -> Dict[str, Any]:
    frames = load_frames(frames_csv)
    imu = load_imu(imu_csv)
    rows: List[Dict[str, Any]] = []
    imu_cursor = 0

    for i, frame in enumerate(frames):
        curr_ns = frame["timestamp_ns"]
        prev_ns = None if i == 0 else frames[i - 1]["timestamp_ns"]

        if prev_ns is None:
            imu_window: List[Dict[str, Any]] = []
        else:
            while imu_cursor < len(imu) and imu[imu_cursor]["timestamp_ns"] <= prev_ns:
                imu_cursor += 1
            j = imu_cursor
            while j < len(imu) and imu[j]["timestamp_ns"] <= curr_ns:
                j += 1
            imu_window = imu[imu_cursor:j]

        integ = integrate_gyro_window(imu_window, prev_ns, curr_ns)
        weight, valid, reason = compute_imu_weight(
            dt=float(integ["dt"]),
            imu_count=int(integ["imu_count"]),
            dr_norm_deg=float(integ["dr_norm_deg"]),
            has_nan=bool(integ["has_nan"]),
        )

        rows.append({
            "frame_id": frame["frame_id"],
            "frame_index": i,
            "timestamp_sec": f"{frame['timestamp_sec']:.9f}",
            "timestamp_ns": frame["timestamp_ns"],
            "filename": f"images/{frame['filename']}",
            "dt": f"{float(integ['dt']):.9f}",
            "imu_count": integ["imu_count"],
            "dr_x": f"{float(integ['dr_x']):.12f}",
            "dr_y": f"{float(integ['dr_y']):.12f}",
            "dr_z": f"{float(integ['dr_z']):.12f}",
            "dr_norm": f"{float(integ['dr_norm']):.12f}",
            "dr_norm_deg": f"{float(integ['dr_norm_deg']):.9f}",
            "dq_w": f"{float(integ['dq_w']):.12f}",
            "dq_x": f"{float(integ['dq_x']):.12f}",
            "dq_y": f"{float(integ['dq_y']):.12f}",
            "dq_z": f"{float(integ['dq_z']):.12f}",
            "imu_valid": valid,
            "imu_weight": f"{weight:.9f}",
            "invalid_reason": reason,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "frame_id", "frame_index", "timestamp_sec", "timestamp_ns", "filename", "dt", "imu_count",
        "dr_x", "dr_y", "dr_z", "dr_norm", "dr_norm_deg", "dq_w", "dq_x", "dq_y", "dq_z",
        "imu_valid", "imu_weight", "invalid_reason"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "frame_count": len(frames),
        "imu_count": len(imu),
        "prior_rows": len(rows),
        "valid_rows": sum(1 for r in rows if int(r["imu_valid"]) == 1),
        "output": str(output_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_dir", type=str, default=None)
    parser.add_argument("--frames", type=str, default=None)
    parser.add_argument("--imu", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.session_dir:
        session_dir = Path(args.session_dir)
        frames_csv = session_dir / "frames.csv"
        imu_csv = session_dir / "imu.csv"
        output_csv = Path(args.output) if args.output else session_dir / "imu_prior.csv"
    else:
        if not args.frames or not args.imu:
            raise SystemExit("Use either --session_dir or both --frames and --imu")
        frames_csv = Path(args.frames)
        imu_csv = Path(args.imu)
        output_csv = Path(args.output) if args.output else frames_csv.parent / "imu_prior.csv"

    result = build_imu_prior(frames_csv, imu_csv, output_csv)
    print(result)


if __name__ == "__main__":
    main()
