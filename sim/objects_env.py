"""Escena de objetos cotidianos con agarre FÍSICO (sin enganche) y PC simulado.

Diferencias con hanoi_env.py:
  - Las mandíbulas colisionan de verdad: el objeto sólo se sostiene por fricción de contacto.
  - Objetos con dimensiones y masas aproximadas (lápiz, móvil, cascos).
  - PC simulado sobre la mesa y zona de entrega "pc_side" junto a él.
  - IK de 5 articulaciones + control del giro de la pinza (para alinear las mandíbulas
    con el objeto: si no, un lápiz no se puede coger por el cuerpo).

Uso (test de agarre, sin VLA):
  ../.venv312/bin/python objects_env.py --grasp-test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from sim_arm import SimArm

HERE = Path(__file__).resolve().parent
OBJECTS_SCENE = HERE / "so101" / "scene_objects.xml"
CONTROL_HZ = 30
GRIP_OPEN = 1.5
GRIP_CLOSED = -0.1745  # límite mecánico: con -0.1 el servo no aprieta (llega a su objetivo sin fuerza)
TRAVEL_Z = 0.12

# Objetos de altura suficiente para que las mandíbulas los pinzen de lado (ver notas de
# Fase 1: lápiz y móvil tumbados quedan por debajo del borde de las mandíbulas).
# pos: (x, y) inicial; grip_dir: dirección horizontal de cierre; grasp_z: altura del TCP al coger;
# rest_z: centro del objeto apoyado en la mesa.
OBJECTS = {
    "bottle": {"pos": (0.17, -0.06), "yaw": 0.0, "grip_dir": (0.0, 1.0), "grasp_z": 0.03, "rest_z": 0.05},
    "box": {"pos": (0.19, 0.04), "yaw": 0.0, "grip_dir": (0.0, 1.0), "grasp_z": 0.02, "rest_z": 0.03},
    "cup": {"pos": (0.16, 0.12), "yaw": 0.0, "grip_dir": (0.0, 1.0), "grasp_z": 0.02, "rest_z": 0.04},
}

# PC simulado: sobre la mesa, con la zona de entrega delante de la pantalla.
PC_POS = (0.12, -0.27)
PC_YAW = 0.0
PC_SIDE_POS = (0.12, -0.19)  # punto donde se deja/entrega el objeto


def _object_xml() -> str:
    """Objetos de sobremesa con dimensiones y masas aproximadas (metros, kg). Freejoint cada uno."""
    return f"""
    <body name="bottle" pos="0 0 0">
      <freejoint name="bottle_free"/>
      <geom name="bottle_geom" type="cylinder" size="0.02 0.05" rgba="0.2 0.6 0.9 1" mass="0.12" friction="1 0.005 0.0001"/>
    </body>
    <body name="box" pos="0 0 0">
      <freejoint name="box_free"/>
      <geom name="box_geom" type="box" size="0.015 0.015 0.03" rgba="0.85 0.6 0.3 1" mass="0.05" friction="1 0.005 0.0001"/>
    </body>
    <body name="cup" pos="0 0 0">
      <freejoint name="cup_free"/>
      <geom name="cup_geom" type="cylinder" size="0.018 0.04" rgba="0.95 0.95 0.95 1" mass="0.06" friction="1 0.005 0.0001"/>
    </body>
    """


def _pc_xml() -> str:
    """PC de sobremesa simplificado: base de teclado + pantalla (cajas), con dimensiones reales."""
    x, y = PC_POS
    return f"""
    <body name="pc" pos="{x} {y} 0" euler="0 0 {PC_YAW}">
      <geom name="pc_base" type="box" size="0.15 0.11 0.01" pos="0 0 0.01" rgba="0.2 0.2 0.22 1"/>
      <geom name="pc_screen" type="box" size="0.15 0.01 0.11" pos="0 -0.09 0.13" rgba="0.05 0.05 0.06 1"/>
      <geom name="pc_screen_face" type="box" size="0.14 0.002 0.10" pos="0 -0.078 0.13" rgba="0.2 0.4 0.7 1"/>
    </body>
    """


def build_scene_xml() -> str:
    return f"""<mujoco model="so101_objects_scene">
    <include file="so101_vla.xml" />

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <global offwidth="640" offheight="480" />
    </visual>

    <worldbody>
        <light pos="0 0 2" dir="0 0 -1" directional="true" />
        <geom name="floor" size="1 1 0.05" pos="0 0 0" type="plane" rgba="0.85 0.85 0.85 1" />

        <!-- Cámaras con los mismos nombres que el checkpoint: top (cenital), wrist (muñeca), table (mesa) -->
        <camera name="top" mode="fixed" pos="0.20 0.00 0.45" xyaxes="0 1 0 -1 0 0" fovy="60" />
        <camera name="table" mode="targetbody" target="base" pos="0.10 -0.80 0.85" fovy="55" />

{_pc_xml()}
{_object_xml()}
    </worldbody>
</mujoco>
"""


def write_scene(path: Path = OBJECTS_SCENE) -> Path:
    path.write_text(build_scene_xml())
    return path


class ObjectsSim(SimArm):
    """SimArm + escena de objetos con colisiones reales en la pinza."""

    def __init__(self, camera_size: tuple[int, int] = (480, 640)):
        write_scene()
        self.obj_qadr: dict[str, int] = {}
        self.held_obj: str | None = None  # objeto que la pinza sostiene (sólo por fricción)
        super().__init__(camera_size=camera_size, scene=OBJECTS_SCENE)
        self.tcp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.jaw_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        for name in OBJECTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            self.obj_qadr[name] = int(self.model.jnt_qposadr[jid])
        self.substeps = max(1, int(round(1.0 / CONTROL_HZ / self.model.opt.timestep)))
        self.close_local = self._closing_axis_local()
        self.reset_objects()

    # --------------------------------------------------------------- escena
    def reset(self) -> dict:
        return self.reset_objects()

    def reset_objects(self, layout: dict | None = None) -> dict:
        """Reinicia brazo y objetos. layout: nombre -> (x, y, yaw); por defecto OBJECTS."""
        if not hasattr(self, "obj_qadr"):  # llamada desde SimArm.__init__
            return SimArm.reset(self)
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = self.home_qpos
        layout = layout or {n: (*o["pos"], o["yaw"]) for n, o in OBJECTS.items()}
        for name, a in self.obj_qadr.items():
            x, y, yaw = layout.get(name, (0.3, 0.3, 0.0))
            z = self._rest_height(name)
            self.data.qpos[a : a + 3] = [x, y, z]
            self.data.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for _ in range(SETTLE_STEPS_OBJ):
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    @staticmethod
    def _rest_height(name: str) -> float:
        return OBJECTS[name]["rest_z"]

    # ------------------------------------------------------------- consultas
    def obj_pos(self, name: str) -> np.ndarray:
        return self.data.qpos[self.obj_qadr[name] : self.obj_qadr[name] + 3].copy()

    def tcp(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    def objects_summary(self) -> dict:
        return {n: [round(float(v), 4) for v in self.obj_pos(n)] for n in OBJECTS}

    # ---------------------------------------------------- cinemática con giro
    def _closing_axis_local(self) -> np.ndarray:
        """Dirección en la que cierran las mandíbulas, expresada en el marco de la pinza.
        Se obtiene midiendo cómo se mueve el centro de masa de la mandíbula móvil al cerrar."""
        saved = self.data.qpos.copy()
        self.data.qpos[5] = GRIP_OPEN
        mujoco.mj_forward(self.model, self.data)
        p_open = self.data.xipos[self.jaw_id].copy()
        self.data.qpos[5] = GRIP_CLOSED
        mujoco.mj_forward(self.model, self.data)
        p_closed = self.data.xipos[self.jaw_id].copy()
        R = self.data.xmat[self.body_id].reshape(3, 3)
        v_world = p_closed - p_open
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        v = R.T @ v_world
        return v / np.linalg.norm(v)

    def _residual(self, q5, target, grip_dir, w_yaw):
        d = self._d
        d.qpos[:5] = q5
        mujoco.mj_forward(self.model, d)
        p = d.site_xpos[self.tcp_site]
        R = d.xmat[self.body_id].reshape(3, 3)
        z_now = R[:, 2]
        v_world = R @ self.close_local
        e_pos = target - p
        e_z = np.array([0.0, 0.0, 1.0]) - z_now
        # Sólo la componente horizontal de la dirección de cierre debe alinearse con grip_dir.
        gd = np.array([grip_dir[0], grip_dir[1], 0.0])
        e_yaw = w_yaw * (v_world[0] * gd[1] - v_world[1] * gd[0])
        return np.concatenate([e_pos, 0.5 * e_z, [e_yaw]])

    def ik(self, target, grip_dir=None, q0=None, iters: int = 120, w_yaw: float = 0.3) -> np.ndarray:
        """DLS con Jacobiano numérico. Controla posición del TCP, pinza hacia abajo y, si se
        pasa grip_dir, el giro de la pinza para cerrar las mandíbulas en esa dirección."""
        self._d = mujoco.MjData(self.model)
        self._d.qpos[:] = self.data.qpos
        if q0 is not None:
            self._d.qpos[:5] = q0
        lo, hi = self.model.jnt_range[:5, 0], self.model.jnt_range[:5, 1]
        q = self._d.qpos[:5].copy()
        use_yaw = grip_dir is not None
        gd = grip_dir if use_yaw else (1.0, 0.0)
        w = w_yaw if use_yaw else 0.0
        eps = 1e-5
        for _ in range(iters):
            r = self._residual(q, target, gd, w)
            if np.linalg.norm(r) < 1e-4:
                break
            J = np.zeros((len(r), 5))
            for i in range(5):
                dq = q.copy()
                dq[i] += eps
                J[:, i] = (self._residual(dq, target, gd, w) - r) / eps
            lam = 1e-3
            step = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(len(r)), -r)
            q = np.clip(q + 0.5 * step, lo, hi)
        return q.copy()

    # ----------------------------------------------------------- agarre físico
    def _tick(self, q5: np.ndarray, grip: float) -> None:
        self.data.ctrl[:6] = np.concatenate([q5, [grip]])
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.sync_view()

    def move_tcp_to(self, target, grip_dir, grip: float, seconds: float, q5_start) -> np.ndarray:
        """Lleva el TCP en línea recta (interpolación suave) con la pinza a `grip`."""
        p0 = self.tcp()
        q5 = np.asarray(q5_start, dtype=float)
        n = max(1, int(round(seconds * CONTROL_HZ)))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            p = p0 + s * (np.asarray(target) - p0)
            q5 = self.ik(p, grip_dir=grip_dir, q0=q5)
            self._tick(q5, grip)
        return q5

    def hold_grip(self, q5, grip: float, seconds: float) -> None:
        """Mantiene la posición y mueve la pinza a `grip` (cerrar o abrir)."""
        n = max(1, int(round(seconds * CONTROL_HZ)))
        for _ in range(n):
            self._tick(q5, grip)

    def grasp_test(self, name: str, reset: bool = True) -> dict:
        """Coge el objeto con la pinza, lo eleva y comprueba que se sostiene sin enganche.
        reset=True reinicia la escena antes (para pruebas); el MCP usa reset=False."""
        o = OBJECTS[name]
        if reset:
            self.reset_objects()
        x, y = self.obj_pos(name)[:2]
        gd = o["grip_dir"]
        q5 = self.data.qpos[:5].copy()
        z_grasp = OBJECTS[name]["grasp_z"]
        q5 = self.move_tcp_to((x, y, TRAVEL_Z), gd, GRIP_OPEN, 2.0, q5)
        q5 = self.move_tcp_to((x, y, z_grasp), gd, GRIP_OPEN, 1.2, q5)
        self.hold_grip(q5, GRIP_CLOSED, 0.8)
        q5 = self.move_tcp_to((x, y, TRAVEL_Z), gd, GRIP_CLOSED, 1.5, q5)
        self.hold_grip(q5, GRIP_CLOSED, 1.0)
        dist = float(np.linalg.norm(self.obj_pos(name) - self.tcp()))
        lifted = float(self.obj_pos(name)[2])
        held = dist < 0.03 and lifted > 0.05
        return {"object": name, "held": bool(held), "obj_z_m": round(lifted, 4),
                "tcp_obj_dist_m": round(dist, 4)}


    # ------------------------------------------------------- API para el MCP
    def move_tcp_cmd(self, x: float, y: float, z: float, gripper: float) -> dict:
        """Mueve la punta de la pinza a (x, y, z) con la pinza hacia abajo y el grado de apertura dado."""
        q5 = self.data.qpos[:5].copy()
        q5 = self.move_tcp_to((x, y, z), None, float(gripper), 1.5, q5)
        self.hold_grip(q5, float(gripper), 0.2)
        return {"tcp": [round(float(v), 4) for v in self.tcp()], "gripper": gripper,
                "held": self.held_obj, "objects": self.objects_summary()}

    def pick(self, name: str) -> dict:
        """Coge el objeto `name` desde su posición actual y lo levanta."""
        if name not in OBJECTS:
            return {"ok": False, "error": f"objeto desconocido: {name}. Disponibles: {list(OBJECTS)}"}
        res = self.grasp_test(name, reset=False)
        self.held_obj = name if res["held"] else None
        res.update({"ok": res["held"], "held_object": self.held_obj, "tcp": [round(float(v), 4) for v in self.tcp()]})
        return res

    def place(self, x: float, y: float, z: float = 0.03) -> dict:
        """Lleva la pinza (con el objeto sujeto) a (x, y, z) y suelta."""
        q5 = self.data.qpos[:5].copy()
        held = self.held_obj
        q5 = self.move_tcp_to((x, y, TRAVEL_Z), None, GRIP_CLOSED, 2.0, q5)
        q5 = self.move_tcp_to((x, y, z), None, GRIP_CLOSED, 1.2, q5)
        self.hold_grip(q5, GRIP_OPEN, 0.8)
        q5 = self.move_tcp_to((x, y, TRAVEL_Z), None, GRIP_OPEN, 1.0, q5)
        self.held_obj = None
        dist = float(np.linalg.norm(self.obj_pos(held) - np.array([x, y, z]))) if held else None
        return {"ok": True, "released": held, "object_xyz": None if held is None else
                [round(float(v), 4) for v in self.obj_pos(held)], "distance_to_target_m": None if dist is None else round(dist, 4)}


SETTLE_STEPS_OBJ = 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp-test", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    sim = ObjectsSim()
    if args.grasp_test:
        names = [args.only] if args.only else list(OBJECTS)
        for n in names:
            print(sim.grasp_test(n), flush=True)


if __name__ == "__main__":
    main()
