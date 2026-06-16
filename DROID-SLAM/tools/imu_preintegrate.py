import os
import argparse
import numpy as np
import pandas as pd


def load_csv(session_dir):
    timestamps_path = os.path.join(session_dir, "timestamps.csv")
    imu_path = os.path.join(session_dir, "imu.csv")

    if not os.path.exists(timestamps_path):
        raise FileNotFoundError(f"timestamps.csv not found: {timestamps_path}")

    if not os.path.exists(imu_path):
        raise FileNotFoundError(f"imu.csv not found: {imu_path}")

    frames = pd.read_csv(timestamps_path)
    imu = pd.read_csv(imu_path)

    required_frame_cols = {"frame_id", "timestamp_sec", "filename"}
    required_imu_cols = {
        "timestamp_sec",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z"
    }

    if not required_frame_cols.issubset(frames.columns):
        raise ValueError(f"timestamps.csv columns invalid: {frames.columns.tolist()}")

    if not required_imu_cols.issubset(imu.columns):
        raise ValueError(f"imu.csv columns invalid: {imu.columns.tolist()}")

    frames = frames.sort_values("timestamp_sec").reset_index(drop=True)
    imu = imu.sort_values("timestamp_sec").reset_index(drop=True)

    return frames, imu


def clean_imu(imu):
    imu = imu.copy()

    numeric_cols = [
        "timestamp_sec",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z"
    ]

    for col in numeric_cols:
        imu[col] = pd.to_numeric(imu[col], errors="coerce")

    imu = imu.dropna(subset=numeric_cols)
    imu = imu.drop_duplicates(subset=["timestamp_sec"])
    imu = imu.sort_values("timestamp_sec").reset_index(drop=True)

    # 비정상적으로 큰 값 제거
    # 스마트폰 IMU 기준으로 넉넉하게 둔 필터
    acc_norm = np.sqrt(imu["acc_x"]**2 + imu["acc_y"]**2 + imu["acc_z"]**2)
    gyro_norm = np.sqrt(imu["gyro_x"]**2 + imu["gyro_y"]**2 + imu["gyro_z"]**2)

    imu = imu[(acc_norm < 100.0) & (gyro_norm < 50.0)].reset_index(drop=True)

    return imu


def integrate_segment(segment):
    """
    frame i-1 ~ frame i 사이의 IMU 데이터를 사전적분한다.

    현재 단계에서는 DROID-SLAM에 직접 넣지 않고,
    데이터 정제 및 motion prior 후보값을 만드는 것이 목적이다.

    출력:
    - dt_total
    - delta_rotation 벡터 근사값
    - delta_velocity 근사값
    - delta_position 근사값

    주의:
    - 이 값은 아직 bias 보정, 중력 정렬, 카메라-IMU extrinsic 보정을 포함하지 않는다.
    - 따라서 최종 pose로 강제 적용하면 안 된다.
    """

    if len(segment) < 2:
        return {
            "dt": 0.0,
            "dr_x": 0.0, "dr_y": 0.0, "dr_z": 0.0,
            "dv_x": 0.0, "dv_y": 0.0, "dv_z": 0.0,
            "dp_x": 0.0, "dp_y": 0.0, "dp_z": 0.0,
            "imu_count": len(segment)
        }

    t = segment["timestamp_sec"].to_numpy(dtype=np.float64)

    acc = segment[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=np.float64)
    gyro = segment[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float64)

    dt_array = np.diff(t)

    # 비정상 dt 제거
    valid = (dt_array > 0.0) & (dt_array < 0.2)

    if valid.sum() == 0:
        return {
            "dt": 0.0,
            "dr_x": 0.0, "dr_y": 0.0, "dr_z": 0.0,
            "dv_x": 0.0, "dv_y": 0.0, "dv_z": 0.0,
            "dp_x": 0.0, "dp_y": 0.0, "dp_z": 0.0,
            "imu_count": len(segment)
        }

    delta_r = np.zeros(3, dtype=np.float64)
    delta_v = np.zeros(3, dtype=np.float64)
    delta_p = np.zeros(3, dtype=np.float64)

    for i, dt in enumerate(dt_array):
        if not valid[i]:
            continue

        # 구간 평균값 사용
        w = 0.5 * (gyro[i] + gyro[i + 1])
        a = 0.5 * (acc[i] + acc[i + 1])

        # 단순 적분
        # 현재는 좌표계/중력 보정 전 단계이므로 motion prior 후보값으로만 사용
        delta_r += w * dt
        delta_p += delta_v * dt + 0.5 * a * dt * dt
        delta_v += a * dt

    dt_total = float(np.sum(dt_array[valid]))

    return {
        "dt": dt_total,
        "dr_x": float(delta_r[0]),
        "dr_y": float(delta_r[1]),
        "dr_z": float(delta_r[2]),
        "dv_x": float(delta_v[0]),
        "dv_y": float(delta_v[1]),
        "dv_z": float(delta_v[2]),
        "dp_x": float(delta_p[0]),
        "dp_y": float(delta_p[1]),
        "dp_z": float(delta_p[2]),
        "imu_count": int(len(segment))
    }


def build_imu_prior(frames, imu):
    rows = []

    for i in range(len(frames)):
        frame_id = int(frames.loc[i, "frame_id"])
        timestamp = float(frames.loc[i, "timestamp_sec"])
        filename = frames.loc[i, "filename"]

        if i == 0:
            rows.append({
                "frame_id": frame_id,
                "frame_index": i,
                "timestamp_sec": timestamp,
                "filename": filename,
                "prev_timestamp_sec": timestamp,
                "dt": 0.0,
                "dr_x": 0.0, "dr_y": 0.0, "dr_z": 0.0,
                "dv_x": 0.0, "dv_y": 0.0, "dv_z": 0.0,
                "dp_x": 0.0, "dp_y": 0.0, "dp_z": 0.0,
                "imu_count": 0
            })
            continue

        prev_t = float(frames.loc[i - 1, "timestamp_sec"])
        curr_t = timestamp

        segment = imu[
            (imu["timestamp_sec"] >= prev_t) &
            (imu["timestamp_sec"] <= curr_t)
        ].copy()

        integ = integrate_segment(segment)

        rows.append({
            "frame_id": frame_id,
            "frame_index": i,
            "timestamp_sec": curr_t,
            "filename": filename,
            "prev_timestamp_sec": prev_t,
            **integ
        })

    return pd.DataFrame(rows)


def print_summary(frames, imu, prior):
    print("========== IMU PREINTEGRATION SUMMARY ==========")
    print(f"frames: {len(frames)}")
    print(f"imu rows after clean: {len(imu)}")
    print(f"prior rows: {len(prior)}")

    if len(frames) > 0:
        print(f"frame time range: {frames['timestamp_sec'].min()} ~ {frames['timestamp_sec'].max()}")

    if len(imu) > 0:
        print(f"imu time range: {imu['timestamp_sec'].min()} ~ {imu['timestamp_sec'].max()}")

    print()
    print("imu_count per frame interval:")
    print(prior["imu_count"].describe())

    print("================================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    session_dir = args.session_dir
    output_path = args.output

    if output_path is None:
        output_path = os.path.join(session_dir, "imu_prior.csv")

    frames, imu = load_csv(session_dir)
    imu = clean_imu(imu)

    prior = build_imu_prior(frames, imu)
    prior.to_csv(output_path, index=False)

    print_summary(frames, imu, prior)
    print(f"[OK] saved imu prior: {output_path}")


if __name__ == "__main__":
    main()