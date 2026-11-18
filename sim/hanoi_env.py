"""Torre de Hanoi en MuJoCo con el SO-101.

Geometría:
  - 3 pivotes fijos (A amarillo, B morado, C naranja) sobre la mesa, en x=0.19.
  - 3 discos anillo (rojo pequeño, verde mediano, azul grande), con agujero central
    para que se puedan apilar sobre el pivote.

Este módulo define la escena (XML generado), el estado lógico (qué disco hay en cada
pivote) y un controlador de IK para pick-and-place, que usan el generador de
demostraciones y el MCP.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from sim_arm import SimArm

HERE = Path(__file__).resolve().parent
HANOI_SCENE = HERE / "so101" / "scene_hanoi.xml"

# Pivotes: letra -> (x, y) en metros, en la zona de trabajo del brazo.
PEGS = {"A": (0.19, -0.12), "B": (0.19, 0.0), "C": (0.19, 0.12)}
PEG_COLOR_NAME = {"A": "yellow", "B": "purple", "C": "orange"}
PEG_RGBA = {"A": "1 0.85 0.1 1", "B": "0.6 0.2 0.8 1", "C": "1 0.5 0.05 1"}
PEG_RADIUS = 0.008
PEG_HEIGHT = 0.06

# Discos: nombre de color -> (radio exterior, tamaño). Tamaño 1 = más pequeño.
DISKS = {
    "red": {"size": 1, "radius": 0.018, "rgba": "0.9 0.1 0.1 1"},
    "green": {"size": 2, "radius": 0.023, "rgba": "0.1 0.7 0.2 1"},
    "blue": {"size": 3, "radius": 0.028, "rgba": "0.1 0.3 0.95 1"},
    "cyan": {"size": 4, "radius": 0.033, "rgba": "0.1 0.85 0.9 1"},
    "brown": {"size": 5, "radius": 0.038, "rgba": "0.45 0.25 0.1 1"},
}
DISK_ORDER = sorted(DISKS, key=lambda c: DISKS[c]["size"])  # de pequeño a grande
MAX_DISKS = len(DISK_ORDER)
DISK_THICK = 0.012
DISK_INNER = 0.011  # radio del agujero
RING_SEGMENTS = 14

GRIPPER_BODIES = ("gripper", "moving_jaw_so101_v1")
TRAVEL_Z = 0.08       # altura de transporte del TCP (m)
GRIP_OPEN = 1.5       # apertura de la pinza (rad)
GRIP_CLOSED = -0.1    # cierre (rad)


def _disk_xml() -> str:
    parts = []
    for color, d in DISKS.items():
        R, rin, h = d["radius"], DISK_INNER, DISK_THICK / 2
        rmid = (R + rin) / 2
        half_rad = (R - rin) / 2
        half_tan = math.pi * (R + rin) / RING_SEGMENTS / 2 * 1.08
        geoms = []
        for k in range(RING_SEGMENTS):
            th = 2 * math.pi * k / RING_SEGMENTS
            x, y = rmid * math.cos(th), rmid * math.sin(th)
            qw, qz = math.cos(th / 2), math.sin(th / 2)
            geoms.append(
                f'<geom type="box" size="{half_rad:.5f} {half_tan:.5f} {h:.5f}" '
                f'pos="{x:.5f} {y:.5f} 0" quat="{qw:.5f} 0 0 {qz:.5f}" '
                f'rgba="{d["rgba"]}" mass="0.004" contype="1" conaffinity="3"/>'
            )
        parts.append(
            f'    <body name="disk_{color}" pos="0 0 0">\n'
            f'      <freejoint name="disk_{color}_free"/>\n'
            + "\n".join("      " + g for g in geoms)
            + "\n    </body>"
        )
    return "\n".join(parts)


def _pegs_xml() -> str:
    out = []
    for letter, (x, y) in PEGS.items():
        out.append(
            f'    <geom name="peg_{letter}" type="cylinder" pos="{x} {y} {PEG_HEIGHT / 2}" '
            f'size="{PEG_RADIUS} {PEG_HEIGHT / 2}" rgba="{PEG_RGBA[letter]}" friction="1 0.005 0.0001" contype="2" conaffinity="2"/>'
        )
        out.append(
            f'    <geom type="cylinder" pos="{x} {y} 0.004" size="0.035 0.004" '
            f'rgba="0.6 0.6 0.6 1" contype="0" conaffinity="0"/>'
        )
    return "\n".join(out)


def build_scene_xml() -> str:
    """Genera el XML de la escena de Hanoi (usa el mismo robot que scene_vla.xml)."""
    # Posiciones iniciales: todos los discos apilados en A (grande abajo).
    x, y = PEGS["A"]
    init = {}
    for i, color in enumerate(["blue", "green", "red"]):
        init[color] = (x, y, DISK_THICK / 2 + i * DISK_THICK + 0.001)
    disk_bodies = _disk_xml()
    # Se fija la posición inicial en el qpos de cada disco al resetear; aquí solo el cuerpo.
    return f"""<mujoco model="so101_hanoi_scene">
    <!-- Escena de Torre de Hanoi generada por sim/hanoi_env.py. No editar a mano. -->
    <include file="so101_vla.xml" />

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <global offwidth="640" offheight="480" />
    </visual>

    <worldbody>
        <light pos="0 0 2" dir="0 0 -1" directional="true" />
        <geom name="floor" size="1 1 0.05" pos="0 0 0" type="plane" rgba="0.85 0.85 0.85 1" />

        <camera name="top" mode="fixed" pos="0.20 0.00 0.45" xyaxes="0 1 0 -1 0 0" fovy="60" />
        <camera name="table" mode="targetbody" target="base" pos="0.10 -0.80 0.85" fovy="55" />

        <!-- Pivotes -->
{_pegs_xml()}

{disk_bodies}
    </worldbody>
</mujoco>
"""


def write_scene(path: Path = HANOI_SCENE) -> Path:
    path.write_text(build_scene_xml())
    return path


class HanoiSim(SimArm):
    """SimArm + escena de Hanoi + estado lógico + IK de pick-and-place."""

    def __init__(self, camera_size: tuple[int, int] = (480, 640)):
        write_scene()  # regenera el XML para que siempre coincida con este módulo
        super().__init__(camera_size=camera_size, scene=HANOI_SCENE)
        self.tcp_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.disk_qadr = {}
        for color in DISKS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"disk_{color}_free")
            self.disk_qadr[color] = int(self.model.jnt_qposadr[jid])
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        # La pinza no colisiona con piezas ni pivotes: el agarre es un enganche simulado (ver hold()).
        for gid in range(self.model.ngeom):
            if self.model.body(self.model.geom_bodyid[gid]).name in GRIPPER_BODIES:
                self.model.geom_contype[gid] = 0
                self.model.geom_conaffinity[gid] = 0
        self.held: str | None = None
        self.active: list[str] = []  # discos en juego; el resto está fuera de la mesa
        self._disk_geoms = {c: [] for c in DISKS}
        for gid in range(self.model.ngeom):
            bname = self.model.body(self.model.geom_bodyid[gid]).name
            if bname.startswith("disk_"):
                self._disk_geoms[bname[len("disk_"):]].append(gid)
        self.reset_hanoi()

    # ------------------------------------------------------------ escena
    def reset_hanoi(self, config: dict[str, list[str]] | None = None, n: int = 3) -> dict:
        """Reinicia el brazo y coloca los discos.

        config: pivote -> lista de colores de abajo a arriba. Si no se pasa, se apilan
        los `n` discos más pequeños en A (el más grande abajo). Los discos que no están
        en `config` se dejan fuera de la mesa, sin colisiones.
        """
        if not hasattr(self, "disk_qadr"):  # llamada desde SimArm.__init__, antes de indexar
            return SimArm.reset(self)
        if config is None:
            if not 1 <= n <= MAX_DISKS:
                raise ValueError(f"n debe estar entre 1 y {MAX_DISKS}")
            config = {"A": list(reversed(DISK_ORDER[:n])), "B": [], "C": []}
        self.active = [c for colors in config.values() for c in colors]
        mujoco.mj_resetData(self.model, self.data)
        self.held = None
        self.data.ctrl[:] = self.home_qpos
        for i, color in enumerate(DISK_ORDER):  # todos fuera; los activos se colocan abajo
            a = self.disk_qadr[color]
            on = color in self.active
            self.data.qpos[a : a + 3] = [0.19, -0.12, 0.0] if on else [-0.3, -0.25 + 0.05 * i, DISK_THICK / 2]
            self.data.qpos[a + 3 : a + 7] = [1, 0, 0, 0]
            for gid in self._disk_geoms[color]:
                self.model.geom_contype[gid] = 1 if on else 0
                self.model.geom_conaffinity[gid] = 3 if on else 0
        for peg, colors in config.items():
            x, y = PEGS[peg]
            for i, color in enumerate(colors):
                a = self.disk_qadr[color]
                self.data.qpos[a : a + 3] = [x, y, DISK_THICK / 2 + i * DISK_THICK + 0.001]
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        return self.get_hanoi_state()

    def reset(self) -> dict:  # el MCP llama a reset() -> escena de Hanoi
        return self.reset_hanoi()

    def hold(self, color: str) -> None:
        """Enganche: el disco pasa a seguir al TCP y deja de colisionar."""
        self.held = color
        for gid in self._disk_geoms[color]:
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0
        self.sync_held()

    def release(self) -> None:
        if self.held is None:
            return
        for gid in self._disk_geoms[self.held]:
            self.model.geom_contype[gid] = 1
            self.model.geom_conaffinity[gid] = 3
        self.held = None

    def grip_update(self, grip_rad: float) -> None:
        """Enganche automático para el VLA: al cerrar la pinza sobre un disco (TCP a <2 cm),
        el disco queda enganchado; al abrir, se suelta. Mismo criterio que el experto."""
        if self.held is None and grip_rad < 0.6:
            tcp = self.tcp()
            best, best_d = None, 0.02
            for color in self.active:
                p = self.disk_xy_z(color)
                d = float(np.linalg.norm(p - tcp))
                if d < best_d:
                    best, best_d = color, d
            if best is not None:
                self.hold(best)
        elif self.held is not None and grip_rad > 0.9:
            self.release()

    def sync_held(self) -> None:
        """Fija la posición del disco agarrado al TCP (llamar en cada subpaso de física)."""
        if self.held is None:
            return
        a = self.disk_qadr[self.held]
        self.data.qpos[a : a + 3] = self.tcp()
        self.data.qpos[a + 3 : a + 7] = [1, 0, 0, 0]
        dof = self.model.jnt_dofadr[self.model.joint(f"disk_{self.held}_free").id]
        self.data.qvel[dof : dof + 6] = 0.0

    def disk_xy_z(self, color: str) -> np.ndarray:
        a = self.disk_qadr[color]
        return self.data.qpos[a : a + 3].copy()

    def top_disk(self, peg: str) -> str | None:
        stack = self.get_hanoi_state()[peg]
        return stack[-1] if stack else None

    def get_hanoi_state(self) -> dict:
        """Estado lógico: pivote -> discos de abajo a arriba (colores)."""
        pegs = {p: [] for p in PEGS}
        for color in self.active:
            p = self.disk_xy_z(color)
            nearest = min(PEGS, key=lambda k: (PEGS[k][0] - p[0]) ** 2 + (PEGS[k][1] - p[1]) ** 2)
            dist = math.dist(PEGS[nearest], p[:2])
            if dist < 0.03:
                pegs[nearest].append((p[2], color))
        return {p: [c for _, c in sorted(v)] for p, v in pegs.items()}

    def get_scene_summary(self) -> dict:
        """Estado del brazo, pivotes y disco agarrado (lo que ve el MCP)."""
        return {"state": self.get_state(), "pegs": self.get_hanoi_state(), "held": self.held}

    def is_solved(self, target: str = "C") -> bool:
        s = self.get_hanoi_state()
        return s[target] == ["blue", "green", "red"]

    # --------------------------------------------------------- cinemática
    def tcp(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    def _approach_dir(self) -> np.ndarray:
        """Eje Z del cuerpo 'gripper' en mundo (debe ser ~(0,0,1) con la pinza hacia abajo)."""
        return self.data.xmat[self.body_id].reshape(3, 3)[:, 2].copy()

    def ik(self, target: np.ndarray, iters: int = 300, q0: np.ndarray | None = None) -> np.ndarray:
        """IK DLS: 5 articulaciones (sin pinza) para que el TCP llegue a `target`
        con la pinza apuntando hacia abajo. Devuelve las 5 posiciones."""
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos
        d.ctrl[:] = self.data.ctrl
        if q0 is not None:
            d.qpos[:5] = q0
        lo, hi = self.model.jnt_range[:5, 0], self.model.jnt_range[:5, 1]
        zd = np.array([0.0, 0.0, 1.0])
        for _ in range(iters):
            mujoco.mj_forward(self.model, d)
            p = d.site_xpos[self.tcp_site]
            zc = d.xmat[self.body_id].reshape(3, 3)[:, 2]
            e_pos = target - p
            e_z = zd - zc
            if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(e_z) < 1e-3:
                break
            mujoco.mj_jacSite(self.model, d, self._jacp, None, self.tcp_site)
            mujoco.mj_jacBody(self.model, d, None, self._jacr, self.body_id)
            Jp = self._jacp[:, :5]
            Jz = -_skew(zc) @ self._jacr[:, :5]  # d(zc)/dq
            J = np.vstack([Jp, 0.5 * Jz])
            e = np.concatenate([e_pos, 0.5 * e_z])
            lam = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            d.qpos[:5] = np.clip(d.qpos[:5] + 0.5 * dq, lo, hi)
        return d.qpos[:5].copy()

def hanoi_instruction(color: str, dst: str) -> str:
    """Instrucción en inglés con la que se entrenó el VLA (ver hanoi_record.py)."""
    return f"Pick up the {color} disk and place it on the {PEG_COLOR_NAME[dst]} peg"


def solve_plan(state: dict[str, list[str]], goal_peg: str = "C") -> list[tuple[str, str, str]]:
    """Solución óptima (BFS) desde `state` hasta dejar todos los discos en `goal_peg`.

    Devuelve una lista de movimientos (disco, origen, destino).
    """
    from collections import deque

    def key(st):
        return tuple(tuple(st[p]) for p in PEGS)

    start = key(state)
    all_disks = sorted((c for v in state.values() for c in v), key=lambda c: -DISKS[c]["size"])
    goal = tuple(tuple(all_disks if p == goal_peg else ())
                 for p in PEGS)
    prev = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        pegs = dict(zip(PEGS, cur))
        for src in PEGS:
            if not pegs[src]:
                continue
            disk = pegs[src][-1]
            for dst in PEGS:
                if dst == src:
                    continue
                if pegs[dst] and DISKS[pegs[dst][-1]]["size"] < DISKS[disk]["size"]:
                    continue
                nxt = {p: list(v) for p, v in pegs.items()}
                nxt[src].pop()
                nxt[dst].append(disk)
                k = key(nxt)
                if k not in prev:
                    prev[k] = (cur, (disk, src, dst))
                    queue.append(k)
    if goal not in prev:
        return []
    moves = []
    cur = goal
    while prev[cur] is not None:
        cur, mv = prev[cur]
        moves.append(mv)
    return list(reversed(moves))


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
