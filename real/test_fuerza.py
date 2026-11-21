"""Prueba LENTA de fuerza del seguidor desde el reposo: cada articulacion sale 7 grados hacia el INTERIOR de su rango y vuelve.

Registra posicion, carga y corriente en cada tick. Nunca sobrepasa la pose inicial hacia el tope (la vuelta termina en ella) ni
el rango de la calibracion menos 3 grados. 1 grado por tick a 10 Hz. Para si no avanza en 15 ticks, si la carga >= 800/1000 o si
la temperatura >= 55 C. Se para con ALLOW_REAL_MOTION=0 o al llegar al tope de tiempo (TEST_FUERZA_MAX_S).

  ./.venv312/bin/python real/test_fuerza.py                      # torque tal como esta (Torque_Limit actual)
  ./.venv312/bin/python real/test_fuerza.py --torque-limit 800   # sube Torque_Limit (RAM, se pierde al apagar) solo para la prueba
  ./.venv312/bin/python real/test_fuerza.py --joints elbow_flex  # solo una articulacion
Al terminar: si el brazo volvio a menos de 2 grados del reposo, APAGA el torque (queda apoyado en el reposo).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MOTION_FILE = ROOT / "real" / ".motion_ok"
CAL = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json"
J5 = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ORDER = ["wrist_flex", "elbow_flex", "shoulder_lift"]
HZ, STEP, OUT_DEG, MARGIN = 10.0, 1.0, 7.0, 3.0


_DEADLINE = time.time() + float(os.environ.get("TEST_FUERZA_MAX_S", "120"))  # tope de tiempo de la prueba


def permiso() -> bool:
    """Sigue mientras no se active la parada (ALLOW_REAL_MOTION=0) y no se pase el tope de tiempo. Con
    CHESS_REQUIRE_PERMISSION=1 exige además el permiso con caducidad de real/authorize_motion.sh."""
    if os.environ.get("ALLOW_REAL_MOTION") == "0" or time.time() > _DEADLINE:
        return False
    if os.environ.get("CHESS_REQUIRE_PERMISSION") != "1":
        return True
    try:
        return time.time() < float(MOTION_FILE.read_text().strip())
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", nargs="*", default=ORDER)
    ap.add_argument("--torque-limit", type=int, default=None)
    ap.add_argument("--breakaway", type=float, default=0.0, help="adelanto extra maximo (grados) que crece 0.4/tick si no avanza. SUBE max_relative_target de LeRobot a 1+este valor")
    a = ap.parse_args()
    if os.environ.get("ALLOW_REAL_MOTION") == "0" or not permiso():
        print("Bloqueado: ALLOW_REAL_MOTION=0 (parada activada), tope de tiempo o falta el permiso pedido con CHESS_REQUIRE_PERMISSION=1.")
        return 3
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cal = json.loads(CAL.read_text())
    lim = {n: (cal[n]["range_max"] - cal[n]["range_min"]) / 2 * 360 / 4095 for n in J5}
    r = SO101Follower(SO101FollowerConfig(port=os.environ["SO101_FOLLOWER_PORT"], id="so101_follower", cameras={}, use_degrees=True,
                                          max_relative_target=STEP + a.breakaway, disable_torque_on_disconnect=False))
    r.connect(calibrate=False)
    bus = r.bus
    if a.torque_limit:
        for m in bus.motors:
            bus.write("Torque_Limit", m, a.torque_limit, normalize=False)
        print(f"Torque_Limit -> {a.torque_limit} (RAM)")

    def pos() -> dict[str, float]:
        o = r.get_observation()
        return {n: float(o[f"{n}.pos"]) for n in J5 + ["gripper"]}

    def tel(j):
        return (int(bus.read("Present_Load", j, normalize=False)), int(bus.read("Present_Current", j, normalize=False)),
                int(bus.read("Present_Temperature", j, normalize=False)))

    start = pos()
    print("pose inicial:", {k: round(v, 1) for k, v in start.items()})
    report = {}
    for j in a.joints:
        s = start[j]
        inward = 1.0 if s < 0 else -1.0
        out = s + inward * OUT_DEG
        if abs(out) > lim[j] - MARGIN:
            print(f"{j}: el objetivo {out:.1f} queda fuera del margen; omito"); continue
        res = {"ida": None, "vuelta": None}
        for fase, goal in (("ida", out), ("vuelta", s)):
            stall, maxload, hist, bias = 0, 0, [], 0.0
            while permiso():
                cur = pos()[j]
                d = goal - cur
                if abs(d) < 0.8:
                    break
                load, cur_ma, temp = tel(j)
                maxload = max(maxload, abs(load))
                hist.append(cur)
                if abs(load) >= 800 or temp >= 55:
                    print(f"{j} {fase}: PARO por carga={load} temp={temp}"); break
                if len(hist) > 15 + int(a.breakaway / 0.4) and abs(hist[-1] - hist[-16]) < 0.3:
                    print(f"{j} {fase}: SIN AVANCE en 15 ticks (pos {cur:.1f}, objetivo {goal:.1f}, carga {load}, corriente {cur_ma})"); break
                if len(hist) > 1 and abs(hist[-1] - hist[-2]) > 0.2:
                    bias = 0.0
                elif a.breakaway > 0:
                    bias = min(a.breakaway, bias + 0.4)
                goal_cmd = cur + float(np.clip(d, -STEP, STEP)) + float(np.sign(d)) * bias
                goal_cmd = float(np.clip(goal_cmd, -(lim[j] - 1.0), lim[j] - 1.0))
                goal_cmd = max(goal_cmd, s) if inward > 0 else min(goal_cmd, s)   # nunca mas alla del reposo hacia el tope
                act = {f"{n}.pos": pos_v for n, pos_v in pos().items()}
                act[f"{j}.pos"] = goal_cmd
                r.send_action(act)
                time.sleep(1 / HZ)
            fin = pos()[j]
            res[fase] = {"llego": abs(goal - fin) < 1.5, "pos_final": round(fin, 1), "objetivo": round(goal, 1), "carga_max": maxload}
            print(f"{j} {fase}: {res[fase]}")
            if not res[fase]["llego"]:
                break
        report[j] = res
        if not (res["ida"] and res["ida"]["llego"] and res["vuelta"] and res["vuelta"]["llego"]):
            print("Paro aqui: no sigo con otras articulaciones."); break
    end = pos()
    print("pose final:", {k: round(v, 1) for k, v in end.items()})
    if all(abs(end[n] - start[n]) < 2.0 for n in J5):
        bus.disable_torque()
        print("Vuelta al reposo: TORQUE APAGADO.")
    else:
        print("NO esta en el reposo: dejo el torque puesto (no suelto el brazo). Colocalo tu o corta con cuidado.")
    r.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
