"""Comprueba el SO-101 real antes de grabar o desplegar. No mueve el brazo.

  ./.venv312/bin/python real/check_hardware.py

Revisa: puertos USB, calibración guardada, cámaras, y lectura de posiciones.
"""

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def ok(msg):
    print(f"[ OK ] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def fail(msg):
    print(f"[FAIL] {msg}")


def main() -> int:
    problems = 0

    ports = sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*")
                   + glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/tty.usbserial*"))
    print("Puertos USB-serie detectados:", ports or "ninguno")
    for label, port, var in [("seguidor", config.FOLLOWER_PORT, "FOLLOWER"), ("líder", config.LEADER_PORT, "LEADER")]:
        if not port:
            warn(f"No hay puerto para el {label} (variable SO101_{var}_PORT).")
            problems += 1
        elif port not in ports:
            fail(f"El puerto del {label} ({port}) no existe ahora mismo.")
            problems += 1
        else:
            ok(f"Puerto del {label}: {port}")

    cal_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"
    for label, id_ in [("seguidor", config.FOLLOWER_ID), ("líder", config.LEADER_ID)]:
        found = list(cal_dir.glob(f"**/{id_}.json")) if cal_dir.exists() else []
        if found:
            ok(f"Calibración del {label} encontrada: {found[0]}")
        else:
            warn(f"No hay calibración del {label} ('{id_}'). Ejecuta real/calibrate.sh o pide la del organizador.")
            problems += 1

    try:
        import cv2
        cams_to_test = [config.TOP_CAM, config.WRIST_CAM] + ([config.TABLE_CAM] if config.TABLE_CAM >= 0 else [])
        for idx in cams_to_test:
            cap = cv2.VideoCapture(idx)
            good = cap.isOpened() and cap.read()[0]
            cap.release()
            (ok if good else fail)(f"Cámara índice {idx}: {'lee imágenes' if good else 'no responde'}")
            problems += 0 if good else 1
    except Exception as e:  # noqa: BLE001
        fail(f"No se pudo probar cámaras: {e}")
        problems += 1

    if config.FOLLOWER_PORT and config.FOLLOWER_PORT in ports:
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
            from lerobot.cameras.opencv import OpenCVCameraConfig

            cams = {
                "top": OpenCVCameraConfig(index_or_path=config.TOP_CAM, width=config.CAM_WIDTH, height=config.CAM_HEIGHT, fps=config.FPS),
                "wrist": OpenCVCameraConfig(index_or_path=config.WRIST_CAM, width=config.CAM_WIDTH, height=config.CAM_HEIGHT, fps=config.FPS),
            }
            robot = SO101Follower(SO101FollowerConfig(port=config.FOLLOWER_PORT, id=config.FOLLOWER_ID, cameras=cams, max_relative_target=config.MAX_RELATIVE_TARGET))
            robot.connect(calibrate=False)
            obs = robot.get_observation()
            joints = {k: round(float(v), 1) for k, v in obs.items() if k.endswith(".pos")}
            ok(f"Lectura del seguidor (sin mover): {joints}")
            robot.disconnect()
        except Exception as e:  # noqa: BLE001
            fail(f"No se pudo conectar al seguidor: {e}")
            problems += 1

    print()
    print("Listo para grabar/desplegar." if problems == 0 else f"Pendiente: {problems} punto(s) por resolver.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
