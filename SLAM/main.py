import os
import csv
import json
import base64
import time
import subprocess
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ============================================================
# FastAPI 기본 설정
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 경로 설정
# ============================================================

SLAM_ROOT = Path("/home/ubuntu/SLAM")
UPLOAD_ROOT = SLAM_ROOT / "uploads"
STATIC_ROOT = SLAM_ROOT / "static"

DROID_ROOT = Path("/home/ubuntu/DROID-SLAM")
DEFAULT_CALIB_PATH = SLAM_ROOT / "default_calib.txt"

# /static/index.html 접근 가능
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


def now_session_id():
    return f"session_{int(time.time() * 1000)}"


# ============================================================
# 요청 모델
# ============================================================

class TriggerRequest(BaseModel):
    session_id: str


class PreintegrateRequest(BaseModel):
    session_id: str


# ============================================================
# 세션 저장 클래스
# ============================================================

class SessionWriter:
    def __init__(self):
        self.session_id = now_session_id()
        self.session_dir = UPLOAD_ROOT / self.session_id
        self.image_dir = self.session_dir / "images"

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.frame_count = 0
        self.imu_count = 0

        self.timestamps_path = self.session_dir / "timestamps.csv"
        self.imu_path = self.session_dir / "imu.csv"
        self.meta_path = self.session_dir / "session_meta.json"
        self.calib_path = self.session_dir / "calib.txt"

        self._init_csv_files()
        self._copy_default_calib()
        self._write_meta()

    def _init_csv_files(self):
        with open(self.timestamps_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "timestamp_sec", "filename"])

        with open(self.imu_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_sec",
                "acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z"
            ])

    def _copy_default_calib(self):
        if DEFAULT_CALIB_PATH.exists():
            with open(DEFAULT_CALIB_PATH, "r") as src:
                calib_text = src.read()

            with open(self.calib_path, "w") as dst:
                dst.write(calib_text)

            print(f"[CALIB] copied default calib to {self.calib_path}")
        else:
            print(f"[CALIB WARNING] default calib not found: {DEFAULT_CALIB_PATH}")
            print("[CALIB WARNING] trigger-slam will fail until calib.txt exists in session folder.")

    def _write_meta(self):
        meta = {
            "session_id": self.session_id,
            "created_at_unix": time.time(),
            "session_dir": str(self.session_dir),
            "image_dir": str(self.image_dir),
            "timestamps_csv": str(self.timestamps_path),
            "imu_csv": str(self.imu_path),
            "calib_txt": str(self.calib_path),
            "default_calib_exists": DEFAULT_CALIB_PATH.exists(),
        }

        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def save_frame_base64(self, timestamp_sec, image_base64):
        filename = f"{self.frame_count:06d}.jpg"
        rel_path = f"images/{filename}"
        save_path = self.image_dir / filename

        if image_base64 is None:
            raise ValueError("image_base64 is None")

        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            raise ValueError(f"invalid base64 image data: {e}")

        if len(image_bytes) == 0:
            raise ValueError("decoded image is empty")

        with open(save_path, "wb") as f:
            f.write(image_bytes)

        with open(self.timestamps_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.frame_count,
                float(timestamp_sec),
                rel_path
            ])

        saved_frame_id = self.frame_count
        self.frame_count += 1

        return saved_frame_id, filename

    def save_frame_binary(self, timestamp_sec, image_bytes):
        filename = f"{self.frame_count:06d}.jpg"
        rel_path = f"images/{filename}"
        save_path = self.image_dir / filename

        if image_bytes is None or len(image_bytes) == 0:
            raise ValueError("binary image is empty")

        with open(save_path, "wb") as f:
            f.write(image_bytes)

        with open(self.timestamps_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.frame_count,
                float(timestamp_sec),
                rel_path
            ])

        saved_frame_id = self.frame_count
        self.frame_count += 1

        return saved_frame_id, filename

    def save_imu(self, data):
        if "timestamp" not in data:
            data["timestamp"] = (
                data.get("timestamp_sec")
                or data.get("time")
                or time.time()
            )

        timestamp_sec = float(data["timestamp"])

        accel = data.get("accel", {})
        gyro = data.get("gyro", {})

        acc_x = data.get("acc_x", data.get("ax", accel.get("x", 0.0)))
        acc_y = data.get("acc_y", data.get("ay", accel.get("y", 0.0)))
        acc_z = data.get("acc_z", data.get("az", accel.get("z", 0.0)))

        gyro_x = data.get("gyro_x", data.get("gx", gyro.get("x", 0.0)))
        gyro_y = data.get("gyro_y", data.get("gy", gyro.get("y", 0.0)))
        gyro_z = data.get("gyro_z", data.get("gz", gyro.get("z", 0.0)))

        row = [
            timestamp_sec,
            float(acc_x),
            float(acc_y),
            float(acc_z),
            float(gyro_x),
            float(gyro_y),
            float(gyro_z),
        ]

        with open(self.imu_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.imu_count += 1


# ============================================================
# 프론트 페이지 제공
# ============================================================

@app.get("/")
def root():
    index_path = STATIC_ROOT / "index.html"

    if not index_path.exists():
        return {
            "status": "running",
            "message": "SLAM server is running, but index.html was not found",
            "expected_index_path": str(index_path),
            "upload_root": str(UPLOAD_ROOT),
            "static_root": str(STATIC_ROOT),
            "droid_root": str(DROID_ROOT),
            "default_calib_path": str(DEFAULT_CALIB_PATH),
            "default_calib_exists": DEFAULT_CALIB_PATH.exists(),
        }

    return FileResponse(index_path)


@app.get("/health")
def health():
    return {
        "status": "running",
        "message": "SLAM server is running",
        "upload_root": str(UPLOAD_ROOT),
        "static_root": str(STATIC_ROOT),
        "index_html": str(STATIC_ROOT / "index.html"),
        "index_exists": (STATIC_ROOT / "index.html").exists(),
        "droid_root": str(DROID_ROOT),
        "default_calib_path": str(DEFAULT_CALIB_PATH),
        "default_calib_exists": DEFAULT_CALIB_PATH.exists(),
    }


# ============================================================
# IMU 사전적분 공통 실행 함수
# ============================================================

def run_imu_preintegration_for_session(session_dir: Path):
    script_path = DROID_ROOT / "tools" / "imu_preintegrate.py"
    timestamps_path = session_dir / "timestamps.csv"
    imu_path = session_dir / "imu.csv"
    output_path = session_dir / "imu_prior.csv"
    log_path = session_dir / "imu_preintegrate_log.txt"

    if not session_dir.exists():
        return {
            "ok": False,
            "message": f"session directory not found: {session_dir}",
            "imu_prior_path": str(output_path),
            "log_path": str(log_path),
            "stderr": ""
        }

    if not script_path.exists():
        return {
            "ok": False,
            "message": f"imu_preintegrate.py not found: {script_path}",
            "imu_prior_path": str(output_path),
            "log_path": str(log_path),
            "stderr": ""
        }

    if not timestamps_path.exists():
        return {
            "ok": False,
            "message": f"timestamps.csv not found: {timestamps_path}",
            "imu_prior_path": str(output_path),
            "log_path": str(log_path),
            "stderr": ""
        }

    if not imu_path.exists():
        return {
            "ok": False,
            "message": f"imu.csv not found: {imu_path}",
            "imu_prior_path": str(output_path),
            "log_path": str(log_path),
            "stderr": ""
        }

    cmd = [
        "python",
        str(script_path),
        "--session_dir",
        str(session_dir)
    ]

    print("[IMU PREINTEGRATE] running:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(DROID_ROOT),
        capture_output=True,
        text=True
    )

    with open(log_path, "w") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(cmd))
        f.write("\n\nSTDOUT:\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)

    if result.returncode != 0:
        return {
            "ok": False,
            "message": "IMU preintegration failed",
            "imu_prior_path": str(output_path),
            "log_path": str(log_path),
            "stderr": result.stderr[-2000:]
        }

    if not output_path.exists():
        return {
            "ok": False,
            "message": "imu_prior.csv was not created",
            "imu_prior_path": str(output_path),
            "log_path": str(log_path),
            "stderr": result.stderr[-2000:]
        }

    return {
        "ok": True,
        "message": "IMU preintegration success",
        "imu_prior_path": str(output_path),
        "log_path": str(log_path),
        "stderr": result.stderr[-2000:]
    }


# ============================================================
# WebSocket: 프레임 + IMU 저장
# ============================================================

@app.websocket("/ws")
@app.websocket("/ws/upload")
@app.websocket("/ws/stream")
@app.websocket("/ws/imu")
@app.websocket("/ws/slam")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session = SessionWriter()
    pending_frame_meta = None

    await websocket.send_json({
        "type": "session_created",
        "session_id": session.session_id,
        "session_dir": str(session.session_dir),
        "calib_exists": session.calib_path.exists(),
    })

    # 기존 프론트 호환용
    await websocket.send_json({
        "type": "server_ready",
        "session_id": session.session_id,
        "session_dir": str(session.session_dir),
    })

    try:
        while True:
            message = await websocket.receive()

            # ------------------------------------------------
            # 1) JSON text 메시지 처리
            # ------------------------------------------------
            if "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"invalid json message: {e}"
                    })
                    continue

                msg_type = (
                    data.get("type")
                    or data.get("kind")
                    or data.get("event")
                )

                # --------------------------------------------
                # start: 기존 프론트 호환용
                # --------------------------------------------
                if msg_type == "start":
                    await websocket.send_json({
                        "type": "start_ack",
                        "ok": True,
                        "session_id": session.session_id,
                        "session_dir": str(session.session_dir),
                        "calib_path": str(session.calib_path),
                        "frames_path": str(session.timestamps_path),
                        "imu_path": str(session.imu_path),
                    })

                # --------------------------------------------
                # frame_meta: 다음 binary 메시지 저장용
                # --------------------------------------------
                elif msg_type in ["frame_meta", "frameMeta"]:
                    timestamp = (
                        data.get("timestamp")
                        or data.get("timestamp_sec")
                        or data.get("time")
                        or time.time()
                    )

                    pending_frame_meta = {
                        "timestamp": float(timestamp),
                        "client_frame_id": data.get("frame_id"),
                        "width": data.get("width"),
                        "height": data.get("height"),
                        "format": data.get("format", "jpeg"),
                        "quality": data.get("quality"),
                        "size_bytes": data.get("size_bytes"),
                    }

                    await websocket.send_json({
                        "type": "frame_meta_ack",
                        "session_id": session.session_id,
                        "client_frame_id": data.get("frame_id"),
                    })

                # --------------------------------------------
                # base64 frame fallback
                # --------------------------------------------
                elif msg_type in ["frame", "image", "video_frame"]:
                    timestamp = (
                        data.get("timestamp")
                        or data.get("timestamp_sec")
                        or data.get("time")
                        or time.time()
                    )

                    image = (
                        data.get("image")
                        or data.get("data")
                        or data.get("frame")
                        or data.get("base64")
                        or data.get("image_base64")
                    )

                    if image is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "frame message requires image/base64 data",
                            "error": "frame message requires image/base64 data",
                        })
                        continue

                    try:
                        saved_frame_id, filename = session.save_frame_base64(timestamp, image)
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"frame save failed: {e}",
                            "error": f"frame save failed: {e}",
                        })
                        continue

                    await websocket.send_json({
                        "type": "frame_saved",
                        "session_id": session.session_id,
                        "frame_id": saved_frame_id,
                        "filename": filename,
                        "mode": "base64",
                        "ok": True,
                        "accepted": True,
                    })

                    # 기존 프론트 호환용
                    await websocket.send_json({
                        "type": "frame_ack",
                        "ok": True,
                        "accepted": True,
                        "session_id": session.session_id,
                        "frame_id": saved_frame_id,
                        "filename": filename,
                        "mode": "base64",
                    })

                # --------------------------------------------
                # IMU 개별 메시지
                # --------------------------------------------
                elif msg_type in ["imu", "sensor", "motion"]:
                    try:
                        session.save_imu(data)

                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"imu save failed: {e}",
                            "error": f"imu save failed: {e}",
                        })

                # --------------------------------------------
                # IMU batch
                # --------------------------------------------
                elif msg_type == "imu_batch":
                    samples = data.get("samples", [])
                    saved = 0

                    for sample in samples:
                        try:
                            session.save_imu(sample)
                            saved += 1
                        except Exception as e:
                            print(f"[IMU BATCH WARNING] failed sample: {e}")

                    await websocket.send_json({
                        "type": "imu_ack",
                        "session_id": session.session_id,
                        "saved": saved,
                        "total_imu_count": session.imu_count
                    })

                # --------------------------------------------
                # 종료: 여기서 자동 사전적분 실행
                # --------------------------------------------
                elif msg_type in ["end", "finish", "stop"]:
                    print(f"[SESSION END] {session.session_id}")
                    print(f"[SESSION END] frames: {session.frame_count}, imu: {session.imu_count}")

                    preintegrate_result = run_imu_preintegration_for_session(session.session_dir)

                    response = {
                        "type": "session_finished",
                        "session_id": session.session_id,
                        "frame_count": session.frame_count,
                        "imu_count": session.imu_count,
                        "session_dir": str(session.session_dir),
                        "calib_exists": session.calib_path.exists(),
                        "preintegrate": preintegrate_result,
                        "imu_prior_path": preintegrate_result.get("imu_prior_path"),
                        "imu_preintegrate_log_path": preintegrate_result.get("log_path"),
                    }

                    await websocket.send_json(response)

                    # 기존 프론트 stop_ack 호환용
                    await websocket.send_json({
                        "type": "stop_ack",
                        "ok": preintegrate_result.get("ok", False),
                        "session_id": session.session_id,
                        "session_dir": str(session.session_dir),
                        "received_frames": session.frame_count,
                        "received_imu": session.imu_count,
                        "calib_path": str(session.calib_path),
                        "frames_path": str(session.timestamps_path),
                        "imu_path": str(session.imu_path),
                        "imu_prior_path": preintegrate_result.get("imu_prior_path"),
                        "preintegrate": preintegrate_result,
                    })

                    break

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"unknown message type: {msg_type}",
                        "error": f"unknown message type: {msg_type}",
                        "received_keys": list(data.keys())
                    })

            # ------------------------------------------------
            # 2) binary JPEG 메시지 처리
            # ------------------------------------------------
            elif "bytes" in message and message["bytes"] is not None:
                image_bytes = message["bytes"]

                if pending_frame_meta is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "received binary image but no pending frame_meta",
                        "error": "received binary image but no pending frame_meta",
                    })
                    continue

                try:
                    saved_frame_id, filename = session.save_frame_binary(
                        pending_frame_meta["timestamp"],
                        image_bytes
                    )

                    await websocket.send_json({
                        "type": "frame_saved",
                        "session_id": session.session_id,
                        "frame_id": saved_frame_id,
                        "client_frame_id": pending_frame_meta.get("client_frame_id"),
                        "filename": filename,
                        "mode": "binary",
                        "size_bytes": len(image_bytes),
                        "ok": True,
                        "accepted": True,
                    })

                    # 기존 프론트 호환용
                    await websocket.send_json({
                        "type": "frame_ack",
                        "ok": True,
                        "accepted": True,
                        "session_id": session.session_id,
                        "frame_id": saved_frame_id,
                        "client_frame_id": pending_frame_meta.get("client_frame_id"),
                        "filename": filename,
                        "mode": "binary",
                        "size_bytes": len(image_bytes),
                    })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"binary frame save failed: {e}",
                        "error": f"binary frame save failed: {e}",
                    })

                finally:
                    pending_frame_meta = None

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "unknown websocket message format",
                    "error": "unknown websocket message format",
                })

    except WebSocketDisconnect:
        print(f"[WS] disconnected: {session.session_id}")

    except Exception as e:
        print(f"[WS ERROR] {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "error": str(e),
            })
        except Exception:
            pass


# ============================================================
# IMU 사전적분 API
# ============================================================

@app.post("/preintegrate-imu")
def preintegrate_imu(req: PreintegrateRequest):
    session_dir = UPLOAD_ROOT / req.session_id

    if not session_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"session not found: {session_dir}"
        )

    result = run_imu_preintegration_for_session(session_dir)

    if not result["ok"]:
        raise HTTPException(
            status_code=500,
            detail=result
        )

    return {
        "status": "success",
        "session_id": req.session_id,
        "imu_prior_path": result["imu_prior_path"],
        "log_path": result["log_path"],
        "message": result["message"]
    }


# ============================================================
# DROID-SLAM baseline 실행 API
# ============================================================

@app.post("/trigger-slam")
def trigger_slam(req: TriggerRequest):
    session_dir = UPLOAD_ROOT / req.session_id
    image_dir = session_dir / "images"
    calib_path = session_dir / "calib.txt"
    reconstruction_path = session_dir / "reconstruction_baseline.pth"

    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="session not found")

    if not image_dir.exists():
        raise HTTPException(status_code=400, detail="images directory not found")

    image_count = len(list(image_dir.glob("*.jpg")))
    if image_count < 10:
        raise HTTPException(
            status_code=400,
            detail=f"not enough images: {image_count}"
        )

    if not calib_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "calib.txt not found in session folder. "
                f"Expected: {calib_path}. "
                f"Create /home/ubuntu/SLAM/default_calib.txt before recording, "
                f"or manually copy calib.txt into this session folder."
            )
        )

    demo_path = DROID_ROOT / "demo.py"
    if not demo_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"DROID-SLAM demo.py not found: {demo_path}"
        )

    cmd = [
        "python",
        "demo.py",
        f"--imagedir={str(image_dir)}",
        f"--calib={str(calib_path)}",
        "--disable_vis",
        f"--reconstruction_path={str(reconstruction_path)}",
    ]

    print("[DROID-SLAM] running:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(DROID_ROOT),
        capture_output=True,
        text=True
    )

    log_path = session_dir / "droid_slam_log.txt"
    with open(log_path, "w") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(cmd))
        f.write("\n\nSTDOUT:\n")
        f.write(result.stdout)
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "DROID-SLAM failed",
                "log_path": str(log_path),
                "stderr": result.stderr[-2000:]
            }
        )

    if not reconstruction_path.exists():
        raise HTTPException(
            status_code=500,
            detail={
                "message": "DROID-SLAM finished but reconstruction file was not created",
                "expected_path": str(reconstruction_path),
                "log_path": str(log_path)
            }
        )

    return {
        "status": "success",
        "session_id": req.session_id,
        "image_count": image_count,
        "reconstruction_path": str(reconstruction_path),
        "log_path": str(log_path)
    }


# ============================================================
# 세션 상태 확인 API
# ============================================================

@app.get("/session/{session_id}")
def get_session_status(session_id: str):
    session_dir = UPLOAD_ROOT / session_id
    image_dir = session_dir / "images"

    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="session not found")

    image_count = 0
    if image_dir.exists():
        image_count = len(list(image_dir.glob("*.jpg")))

    def line_count(path: Path):
        if not path.exists():
            return 0
        with open(path, "r", errors="ignore") as f:
            return sum(1 for _ in f)

    files = {
        "timestamps_csv": (session_dir / "timestamps.csv").exists(),
        "imu_csv": (session_dir / "imu.csv").exists(),
        "calib_txt": (session_dir / "calib.txt").exists(),
        "session_meta_json": (session_dir / "session_meta.json").exists(),
        "imu_prior_csv": (session_dir / "imu_prior.csv").exists(),
        "imu_preintegrate_log_txt": (session_dir / "imu_preintegrate_log.txt").exists(),
        "reconstruction_baseline_pth": (session_dir / "reconstruction_baseline.pth").exists(),
        "droid_slam_log_txt": (session_dir / "droid_slam_log.txt").exists(),
    }

    return {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "image_count": image_count,
        "timestamps_rows": max(0, line_count(session_dir / "timestamps.csv") - 1),
        "imu_rows": max(0, line_count(session_dir / "imu.csv") - 1),
        "imu_prior_rows": max(0, line_count(session_dir / "imu_prior.csv") - 1),
        "files": files,
    }