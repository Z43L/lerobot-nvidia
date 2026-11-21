"""Verificación de principio a fin del ajedrez real SO-101, peldaños 0-4. NO mueve el brazo y no pide permisos.

  ./.venv312/bin/python pipeline/e2e_check.py                 # todo
  ./.venv312/bin/python pipeline/e2e_check.py --stage 2       # un peldaño
  ./.venv312/bin/python pipeline/e2e_check.py --camera 0      # además lee el tablero en vivo (peldaño 2)
  ./.venv312/bin/python pipeline/e2e_check.py --json out.json

Cada comprobación termina en PASS, FAIL o SKIP (SKIP = no se puede comprobar aquí, con el motivo). Un SKIP NO es un PASS:
el informe final lo cuenta aparte. Código de salida 0 solo si no hay FAIL.

  0 Entorno      paquetes, tests, configuración
  1 Hardware     puertos, calibración del seguidor, cámaras (solo lectura)
  2 Visión       pesos, esquinas, lectura de fotos y de la cámara en vivo
  3 Mapa         calibración del tablero, origen, cinemática: ¿llega a las 64 casillas? (sin mover)
  models ACT       ¿cada checkpoint carga y predice en este dispositivo? error offline contra el dataset
  4 Ensayo       partida en dry por la CLI y por el MCP (sin mover)
Los peldaños 5-8 (mueven el brazo) NO están aquí: esta prueba nunca mueve el brazo (fuerza ALLOW_REAL_MOTION=0).
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
for sub in ("hub", "sim", "vision", "real"):
    sys.path.insert(0, str(ROOT / sub))

import chess_config as CFG  # noqa: E402  (carga real/env.local.sh)

RESULTS: list[dict] = []
# Este script es una prueba SIN movimiento: activa la parada (ALLOW_REAL_MOTION=0) y el modo con permiso en todo lo que
# lanza, sea cual sea la política del repo. Si aun así el gate dejara mover, la prueba del gate no se lanza.
SAFE_ENV = {"ALLOW_REAL_MOTION": "0", "CHESS_REQUIRE_PERMISSION": "1", "CHESS_ARM": "none"}
os.environ.update(SAFE_ENV)


def record(stage: int, name: str, status: str, detail: str = "", criterion: str = "") -> str:
    RESULTS.append({"stage": stage, "check": name, "status": status, "detail": detail, "criterion": criterion})
    mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️ "}[status]
    print(f"  {mark} {name}" + (f"  — {detail}" if detail else ""))
    return status


def run(cmd: list[str], env_extra: dict | None = None, timeout: int = 180) -> subprocess.CompletedProcess:
    env = dict(os.environ, **(env_extra or {}), **SAFE_ENV)  # este script nunca debe poder mover el brazo
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env, timeout=timeout)


# ------------------------------------------------------------------ 0 entorno
def stage0():
    print("\n[0] Entorno")
    need = {"torch": True, "lerobot": True, "cv2": True, "chess": True, "mcp": True, "serial": True, "numpy": True,
            "mujoco": False, "ultralytics": True}
    for mod, required in need.items():
        try:
            m = importlib.import_module(mod)
            record(0, f"paquete {mod}", "PASS", getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            record(0, f"paquete {mod}", "FAIL" if required else "SKIP", f"{type(e).__name__}: {e}"[:90],
                   "importable")
    try:
        import torch
        ok = torch.backends.mps.is_available()
        record(0, "MPS (GPU de Apple)", "PASS" if ok else "FAIL", "disponible" if ok else "no disponible: iría a CPU (31 s/chunk)")
    except Exception:  # noqa: BLE001
        pass
    r = run([PY, "sim/test_chess_real.py"])
    n_ok = r.stdout.count("OK ")
    record(0, "tests de ajedrez real", "PASS" if r.returncode == 0 else "FAIL",
           f"{n_ok} OK" + ("" if r.returncode == 0 else f"; {r.stderr.strip().splitlines()[-1][:90] if r.stderr.strip() else ''}"),
           "8 pruebas OK")
    record(0, "límite de paso", "PASS" if CFG.STEP_DEG <= 2.0 else "FAIL",
           f"{CFG.STEP_DEG}°/paso", "<= 2 (AGENTS.md)")


# ------------------------------------------------------------------ 1 hardware
def stage1(args):
    print("\n[1] Hardware (solo lectura)")
    ports = sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*"))
    record(1, "puertos USB serie", "PASS" if ports else "SKIP", ", ".join(ports) or "ninguno: este equipo no tiene el brazo", ">= 1 (seguidor)")
    fp = CFG.FOLLOWER_PORT
    if not fp:
        record(1, "SO101_FOLLOWER_PORT configurado", "FAIL", "falta en entorno y en real/env.local.sh", "definido")
    else:
        record(1, "puerto del seguidor existe", "PASS" if Path(fp).exists() else ("SKIP" if not ports else "FAIL"),
               fp if Path(fp).exists() else f"{fp} no existe ahora mismo", "el fichero existe")
    cal = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{CFG.FOLLOWER_ID}.json"
    record(1, "calibración del seguidor", "PASS" if cal.exists() else "SKIP",
           str(cal) if cal.exists() else f"no existe {cal} (solo está en el host)", "existe")
    for label, idx in (("cámara de YOLO", CFG.YOLO_CAM), ("cámara de muñeca", CFG.WRIST_CAM)):
        record(1, f"{label} (índice {idx})", *_camera_check(idx))


def _camera_check(idx: int):
    import cv2
    cap = cv2.VideoCapture(idx)
    ok, frame = (False, None)
    if cap.isOpened():
        import time
        for _ in range(40):  # las cámaras de macOS tardan en dar el primer fotograma
            ok, frame = cap.read()
            if ok and frame is not None and frame.std() > 1.0:
                break
            time.sleep(0.1)
    cap.release()
    if not ok:
        return "FAIL", f"el índice {idx} no abre o no da imagen (¿permiso de cámara de macOS?)", "devuelve imagen"
    h, w = frame.shape[:2]
    return "PASS", f"{w}x{h}", "devuelve imagen"


# ------------------------------------------------------------------ 2 visión
def stage2(args):
    print("\n[2] Visión")
    record(2, "pesos de YOLO", "PASS" if CFG.YOLO_WEIGHTS.exists() else "FAIL", str(CFG.YOLO_WEIGHTS.relative_to(ROOT)) if CFG.YOLO_WEIGHTS.exists() else f"no existe {CFG.YOLO_WEIGHTS}", "existe")
    record(2, "esquinas del tablero en píxeles", "PASS" if CFG.BOARD_GEOMETRY.exists() else "FAIL", CFG.BOARD_GEOMETRY.name, "existe")
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        record(2, "lectura de fotos con YOLO", "SKIP", "falta ultralytics en este entorno", "recall >= 0,9")
        if args.camera is not None:
            record(2, "lectura del tablero en vivo", "SKIP", "falta ultralytics", "FEN inicial, <= 3 dudosas")
        return
    import chess
    import cv2
    from chess_vision import ChessVision, verify_move

    cv = ChessVision()
    # 2a. fotos de validación con posición conocida
    pos_dir = ROOT / "vision" / "dataset_camara" / "positions"
    files = sorted(pos_dir.glob("*.json")) if pos_dir.exists() else []
    if not files:
        record(2, "validación con fotos etiquetadas", "SKIP", f"no hay {pos_dir}", "recall >= 0,9")
    else:
        tp = fp = fn = exact = n = 0
        for f in files:
            meta = json.loads(f.read_text())
            fen = meta.get("fen") if isinstance(meta, dict) else None
            img = next(iter((ROOT / "vision/dataset_camara/images").glob(f"*/{f.stem}.*")), None)
            if not fen or img is None:
                continue
            exp = ChessVision.board_to_squares(chess.Board(fen + " w - - 0 1" if " " not in fen else fen))
            seen = cv.read_squares(cv2.imread(str(img)))
            got = {s: v[:2] for s, v in seen.items()}
            tp += sum(1 for s in exp if got.get(s) == exp[s])
            fn += sum(1 for s in exp if got.get(s) != exp[s])
            fp += sum(1 for s in got if s not in exp)
            exact += got == exp
            n += 1
        if n == 0:
            record(2, "validación con fotos etiquetadas", "SKIP", "formato de positions/ no reconocido", "recall >= 0,9")
        else:
            recall, prec = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
            record(2, "validación con fotos etiquetadas", "PASS" if recall >= 0.9 and prec >= 0.9 else "FAIL",
                   f"{n} fotos: recall {recall:.2f}, precisión {prec:.2f}, exactas {exact}/{n}", "recall y precisión >= 0,90")
    # 2b. la verificación de jugada rechaza lo que debe
    b = chess.Board()
    m = chess.Move.from_uci("e2e4")
    a = b.copy(); a.push(m)
    pos = {s: (c, k, 0.9) for s, (c, k) in ChessVision.board_to_squares(a).items()}
    neg = {s: (c, k, 0.9) for s, (c, k) in ChessVision.board_to_squares(b).items()}
    okp, okn = verify_move(b, m, pos)[0], verify_move(b, m, neg)[0]
    record(2, "verify_move: acepta la jugada hecha y rechaza la no hecha", "PASS" if okp and not okn else "FAIL",
           f"hecha={okp}, no hecha={okn}", "True / False")
    # 2c. cámara en vivo
    if args.camera is None:
        record(2, "lectura del tablero en vivo", "SKIP", "usa --camera N con el tablero en posición inicial", "FEN inicial, <= 3 dudosas")
    else:
        seen = cv.read_squares(__import__("chess_vision").capture(args.camera))
        exp = ChessVision.board_to_squares(chess.Board())
        bad = [s for s in sorted(set(exp) | set(seen)) if exp.get(s) != (seen[s][:2] if s in seen else None)]
        record(2, "lectura del tablero en vivo", "PASS" if len(bad) <= 3 else "FAIL",
               f"{len(bad)} casillas distintas de la posición inicial: {bad[:8]}", "<= 3 casillas dudosas")


# ------------------------------------------------------------------ 3 mapa
def stage3(args):
    print("\n[3] Mapa del tablero y cinemática (sin mover)")
    board_p = ROOT / "hub" / "chess_board_real.json"
    if not board_p.exists():
        record(3, "calibración del tablero real", "FAIL", "falta hub/chess_board_real.json (skill so101-calibracion-tablero)", "existe")
        return
    d = json.loads(board_p.read_text())
    cal = d.get("calibration", {})
    record(3, "calibración real", "PASS" if d.get("status") == "real" else "FAIL", f"status={d.get('status')}, {cal.get('date', '?')}", "status=real")
    rms = cal.get("rms_mm")
    record(3, "error de ajuste (RMS)", "PASS" if rms is not None and rms <= 5 else "FAIL", f"{rms} mm", "<= 5 mm")
    loo = cal.get("leave_one_out_mm", {})
    worst = max(loo.values()) if loo else None
    record(3, "validación cruzada (peor toque)", "PASS" if worst is not None and worst <= 12 else "FAIL",
           f"{worst} mm (~1/3 de casilla = {d['square_size_m'] * 1000 / 3:.0f} mm)", "<= 12 mm")
    sq = d["square_size_m"] * 1000
    record(3, "tamaño de casilla", "PASS" if 25 <= sq <= 40 else "FAIL", f"{sq:.1f} mm", "25-40 mm")
    record(3, "pose de origen", "PASS" if (ROOT / "hub/chess_home.json").exists() else "FAIL", "hub/chess_home.json", "existe")
    calf = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{CFG.FOLLOWER_ID}.json"
    if not calf.exists():
        record(3, "alcance IK de las 64 casillas", "SKIP", "falta la calibración del seguidor (solo en el host)", "64/64 con margen >= 1°")
        return
    import chess
    from chess_ik_exec import GRASP_DZ, Planner

    pl = Planner()
    unreachable, tight = [], []
    for name, (x, y) in d["squares"].items():
        sol = pl.kin.solve_xy((x, y), GRASP_DZ[chess.PAWN])
        if sol["joints"] is None:
            unreachable.append(name)
        elif sol["margin"] is not None and sol["margin"] < 1.0:
            tight.append(name)
    record(3, "alcance IK de las 64 casillas (peón)", "PASS" if not unreachable and not tight else "FAIL",
           f"sin solución: {unreachable or 'ninguna'}; a <1° de un límite: {tight or 'ninguna'}", "64/64 con margen >= 1°")


# ------------------------------------------------------------------ 4 ensayo
def stage4(args):
    print("\n[4] Ensayo en dry (sin mover)")
    tmp = Path(tempfile.mkdtemp(prefix="e2e_"))
    env = {"CHESS_GAME_FILE": str(tmp / "g.json"), "CHESS_BACKEND": "real", "CHESS_REAL_EXECUTOR": "dry",
           "CHESS_REAL_PLAN_LOG": str(tmp / "plan.json")}
    cli = [PY, "sim/chess_cli.py"]
    steps = [cli + ["new", "--robot-white"], cli + ["robot", "--seed", "3"]]
    ok = all(run(c, env).returncode == 0 for c in steps[:1])
    # partida corta: robot, humano (primera legal), robot, humano...
    import chess
    plies = 0
    for i in range(6):
        g = json.loads((tmp / "g.json").read_text())
        b = chess.Board(g["fen"])
        robot_turn = (b.turn == chess.WHITE) == (g["robot_color"] == "white")
        r = run(cli + (["robot", "--seed", str(i)] if robot_turn else ["human", next(iter(b.legal_moves)).uci()]), env)
        if r.returncode != 0:
            ok = False
            record(4, f"CLI dry, ply {i + 1}", "FAIL", (r.stdout + r.stderr).strip()[-120:], "código 0")
            break
        plies += 1
    g = json.loads((tmp / "g.json").read_text()) if (tmp / "g.json").exists() else {"history": []}
    consistent = len(g["history"]) == plies and ok
    record(4, "partida por la CLI (backend real, dry)", "PASS" if consistent else "FAIL", f"{plies} jugadas registradas", "6 jugadas, historial coherente")
    plan = json.loads((tmp / "plan.json").read_text()) if (tmp / "plan.json").exists() else {}
    record(4, "plan en metros de la calibración real", "PASS" if plan.get("geometria_real") else "FAIL",
           f"geometría {plan.get('geometria')}", "geometria_real=true")
    # sim
    senv = {"CHESS_GAME_FILE": str(tmp / "s.json"), "CHESS_BACKEND": "sim"}
    try:
        import mujoco  # noqa: F401
        a = run(cli + ["new"], senv).returncode == 0 and run(cli + ["human", "e4"], senv).returncode == 0
        a = a and run(cli + ["robot", "--seed", "1"], senv, timeout=240).returncode == 0
        record(4, "backend sim (regresión)", "PASS" if a else "FAIL", "e4 + respuesta del robot", "ambas jugadas")
    except ImportError:
        record(4, "backend sim (regresión)", "SKIP", "falta mujoco", "ambas jugadas")
    # MCP
    asyncio.run(_mcp_check(tmp))


async def _mcp_check(tmp: Path):
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        record(4, "MCP so101-chess", "SKIP", "falta mcp", "")
        return
    env = dict(os.environ)
    env.update(SAFE_ENV)
    env.update(CHESS_ARM="none", CHESS_GAME_FILE=str(tmp / "m.json"), CHESS_REAL_PLAN_LOG=str(tmp / "mp.json"))
    try:
        async with stdio_client(StdioServerParameters(command=PY, args=["hub/chess_mcp.py"], env=env, cwd=str(ROOT))) as (r, w):
            async with ClientSession(r, w) as s:
                await asyncio.wait_for(s.initialize(), 60)
                tools = {t.name for t in (await s.list_tools()).tools}
                need = {"chess_move", "chess_plan", "chess_commit", "chess_verify_move", "chess_see", "arm_stop"}
                record(4, "MCP: herramientas registradas", "PASS" if need <= tools else "FAIL", f"{len(tools)}; faltan {sorted(need - tools) or 'ninguna'}", "todas las esperadas")

                async def call(name, **a):
                    res = await s.call_tool(name, a)
                    t = res.content[0].text if res.content else ""
                    try:
                        return json.loads(t)
                    except ValueError:
                        return t
                await call("chess_new", robot_white=True)
                dry = await call("chess_move", uci="e2e4", executor="dry")
                record(4, "MCP: chess_move dry", "PASS" if dry.get("status") == "ok" and dry.get("verificada") is False else "FAIL", f"{dry.get('status')}", "ok, sin mover")
                ill = await call("chess_move", uci="e2e5", executor="dry")
                record(4, "MCP: rechaza jugada ilegal", "PASS" if ill.get("status") == "refused" else "FAIL", f"{ill.get('status')}", "refused")
                ok_gate, why_gate = CFG.motion_permission()
                if ok_gate:  # el gate permitiría mover: NO se lanza la prueba (movería el brazo de verdad)
                    record(4, "MCP: la verificación no mueve el brazo (parada forzada)", "FAIL",
                           f"motion_permission() deja mover incluso con CHESS_REQUIRE_PERMISSION=1 y sin ALLOW_REAL_MOTION ({why_gate}); prueba NO lanzada",
                           "blocked (nunca intenta mover)")
                    await call("chess_commit", uci="e2e4")
                    st = await call("chess_state")
                    record(4, "MCP: registrar tras verificar", "PASS" if isinstance(st, str) and "Historial: e4" in st else "FAIL", "historial con e4", "e4 registrada")
                    return
                blk = await call("chess_move", uci="e2e4", executor="ik")
                record(4, "MCP: la verificación no mueve el brazo (parada forzada)", "PASS" if blk.get("status") == "blocked" else "FAIL",
                       f"{blk.get('status')}: {str(blk.get('message', ''))[:60]}", "blocked con ALLOW_REAL_MOTION=0")
                await call("chess_commit", uci="e2e4")
                st = await call("chess_state")
                record(4, "MCP: registrar tras verificar", "PASS" if isinstance(st, str) and "Historial: e4" in st else "FAIL", "historial con e4", "e4 registrada")
    except Exception as e:  # noqa: BLE001
        record(4, "MCP so101-chess", "FAIL", f"{type(e).__name__}: {e}"[:110], "arranca y responde")



# ------------------------------------------------------------------ models (ACT)
def stage_models(args):
    print("\n[models] ACT genérico: ¿carga y predice en ESTE dispositivo? (un checkpoint de CUDA debe cargar en un Mac)")
    import numpy as np
    import time
    try:
        import torch
    except ImportError:
        record(5, "torch", "FAIL", "no importable")
        return
    device = os.environ.get("SO101_VLA_DEVICE", "mps")
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    sys.path.insert(0, str(ROOT / "real"))
    found = {n: p for n, p in CFG.ACT_MODELS.items() if (p / "config.json").exists()}
    for n, p in CFG.ACT_MODELS.items():
        if n not in found:
            record(5, f"modelo {n}", "SKIP", f"no está en {p} (cópialo con rsync, pipeline/MODELOS_ACT.md)", "existe")
    if not found:
        record(5, "algún ACT disponible", "FAIL", "ningún checkpoint en las rutas conocidas ni en CHESS_ACT_CHECKPOINT", ">= 1")
        return
    from deploy_act_generic import build_obs
    from deploy_goal import load_policy, step_policy
    motors = [f"{k}.pos" for k in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
    frames = {k: np.zeros((360, 640, 3), np.uint8) for k in ("front", "gripper")}
    dataset = Path(os.environ.get("ACT_EVAL_DATASET", Path.home() / "data-play-tray-1-clean"))
    for name, path in found.items():
        weights = path / "model.safetensors"
        if weights.stat().st_size < 1_000_000:
            record(5, f"{name}: pesos", "FAIL", f"model.safetensors de {weights.stat().st_size} bytes (puntero LFS)", "> 1 MB")
            continue
        try:
            pol, pre, post = load_policy(path, device)
            record(5, f"{name}: carga en {device}", "PASS", f"entrenado en {json.loads((path / 'config.json').read_text()).get('device')}", "carga sin error")
        except Exception as e:  # noqa: BLE001
            record(5, f"{name}: carga en {device}", "FAIL", f"{type(e).__name__}: {str(e)[:100]}", "carga sin error")
            continue
        if "observation.environment_state" in pol.config.input_features:
            record(5, f"{name}: tipo", "SKIP", "es un ACT con objetivo: no sirve para el ejecutor ik", "sin objetivo")
            continue
        state = np.array([0, 0, 0, 0, 167, 0], np.float32)
        step_policy(pol, pre, post, build_obs(state, frames), device, motors)  # calentamiento
        pol.reset()
        t = time.time()
        act = step_policy(pol, pre, post, build_obs(state, frames), device, motors)
        dt = time.time() - t
        vals = np.array([act[m] for m in motors])
        sane = bool(np.isfinite(vals).all() and 0 <= vals[5] <= 100 and np.abs(vals[:5]).max() < 200)
        record(5, f"{name}: predicción", "PASS" if sane else "FAIL", f"{dt:.2f} s/chunk, gripper {vals[5]:.1f}", "finita, pinza 0-100")
        try:
            import chess_ik_exec as ikx
            lo, hi = ikx.trained_gripper_range(path)
            approach = getattr(ikx, "GRIP_APPROACH", None)  # apertura de acercamiento por pieza (la de tránsito, 40, no la ve el ACT)
            grip = max(approach.values()) if approach else ikx.GRIP_OPEN
            why = ikx.act_grip_problem(path, grip_open=grip)
            src = "máx. de GRIP_APPROACH por pieza" if approach else "GRIP_OPEN"
            record(5, f"{name}: apertura de aproximación dentro de lo entrenado", "FAIL" if why else "PASS",
                   why or f"{src}={grip:.0f} dentro de {lo:.1f}-{hi:.1f}", "apertura de acercamiento <= máx. entrenado + 2")
        except Exception as e:  # noqa: BLE001
            record(5, f"{name}: apertura de aproximación", "FAIL", f"{type(e).__name__}: {e}"[:100], "comprobable")
        if not (dataset / "meta/info.json").exists():
            record(5, f"{name}: error offline", "SKIP", f"no hay dataset en {dataset} (ACT_EVAL_DATASET)", "<= 5° de media")
            continue
        r = run([PY, "pipeline/offline_eval.py", str(path), str(dataset), "--episodes", "0,10,30,45,58", "--stride", "25", "--device", device], timeout=600)
        errs = [float(l.split("error medio")[1].split("°")[0]) for l in r.stdout.splitlines() if "error medio" in l]
        if not errs:
            record(5, f"{name}: error offline", "FAIL", (r.stderr.strip().splitlines() or ["sin salida"])[-1][:100], "<= 5° de media")
        else:
            m = sum(errs) / len(errs)
            record(5, f"{name}: error offline (EN MUESTRA)", "PASS" if m <= 5 else "FAIL",
                   f"{m:.1f}° de media en {len(errs)} episodios; entrenado con estos episodios: mide ajuste, no generalización", "<= 5°")

# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["0", "1", "2", "3", "4", "models"])
    ap.add_argument("--camera", type=int, default=None, help="índice de la cámara que ve el tablero (peldaño 2, en vivo)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    stages = {"0": lambda: stage0(), "1": lambda: stage1(args), "2": lambda: stage2(args), "3": lambda: stage3(args),
              "models": lambda: stage_models(args), "4": lambda: stage4(args)}
    for k in ([args.stage] if args.stage is not None else list(stages)):
        stages[k]()
    c = {s: sum(1 for r in RESULTS if r["status"] == s) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\nRESUMEN: {c['PASS']} PASS, {c['FAIL']} FAIL, {c['SKIP']} SKIP (un SKIP no está comprobado)")
    if c["FAIL"]:
        print("FALLAN:", "; ".join(f"[{r['stage']}] {r['check']}" for r in RESULTS if r["status"] == "FAIL"))
    if args.json:
        Path(args.json).write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False))
    return 1 if c["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
