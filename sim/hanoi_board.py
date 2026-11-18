"""Tablero de la torre de Hanoi, con la misma interfaz en simulación y en el brazo real.

Claude orquesta: pide el estado, elige el siguiente movimiento y llama a `pick_place`, que
ejecuta el VLA de ese color (uno por color). La diferencia es quién conoce el estado:

- SimBoard: la simulación sabe dónde está cada disco. El resultado se comprueba solo.
- RealBoard: no hay sensores de disco. El estado es lógico y se guarda en real/hanoi_state.json.
  Tras cada movimiento el MCP devuelve las cámaras y espera a que Claude (o una persona) confirme
  con `confirm(ok)`. Si el disco no quedó donde debía, se corrige con `set_state`.
"""

from __future__ import annotations

import json
from pathlib import Path

from hanoi_env import DISK_ORDER, DISKS, PEGS, hanoi_instruction

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "real" / "hanoi_state.json"


def default_pegs(n: int) -> dict[str, list[str]]:
    return {"A": list(reversed(DISK_ORDER[:n])), "B": [], "C": []}


def check_move(pegs: dict[str, list[str]], disk: str, to_peg: str) -> tuple[str | None, str | None]:
    """Devuelve (pivote_origen, error). Error si el movimiento no es legal."""
    if disk not in DISKS or to_peg not in PEGS:
        return None, "disco o pivote no válido"
    src = next((p for p in PEGS if pegs[p][-1:] == [disk]), None)
    if src is None:
        return None, f"{disk} no está encima de ningún pivote"
    if src == to_peg:
        return src, "ya está en ese pivote"
    if pegs[to_peg] and DISKS[pegs[to_peg][-1]]["size"] < DISKS[disk]["size"]:
        return src, f"no se puede poner {disk} sobre {pegs[to_peg][-1]}"
    return src, None


def _valid_pegs(pegs: dict[str, list[str]]) -> str | None:
    """Comprueba que la configuración sea legal (sin discos repetidos ni apilados sobre menores)."""
    seen = [c for v in pegs.values() for c in v]
    if len(seen) != len(set(seen)) or any(c not in DISKS for c in seen):
        return "discos repetidos o desconocidos"
    for stack in pegs.values():
        for below, above in zip(stack, stack[1:]):
            if DISKS[above]["size"] > DISKS[below]["size"]:
                return f"{above} sobre {below}: un disco no puede ir sobre uno más pequeño"
    unknown = set(pegs) - set(PEGS)
    return f"pivotes desconocidos: {sorted(unknown)}" if unknown else None


class SimBoard:
    """Tablero sobre HanoiSim. `vla_for(color)` devuelve el VLARunner de ese color."""

    kind = "sim"

    def __init__(self, arm, vla_for):
        self.arm = arm
        self.vla_for = vla_for

    def state(self) -> dict:
        return {"pegs": self.arm.get_hanoi_state(), "held": self.arm.held,
                "disks": sorted(self.arm.active, key=DISK_ORDER.index)}

    def reset(self, n: int = 3) -> dict:
        self.arm.reset_hanoi(n=n)
        return self.state()

    def pick_place(self, disk: str, to_peg: str, max_steps: int = 600) -> dict:
        pegs = self.arm.get_hanoi_state()
        if disk not in self.arm.active:
            return {"ok": False, "error": f"{disk} no está en la torre actual"}
        src, err = check_move(pegs, disk, to_peg)
        if err:
            return {"ok": src is not None and src == to_peg, "error": err, "pegs": pegs}

        def done() -> bool:
            return self.arm.held is None and self.arm.get_hanoi_state()[to_peg][-1:] == [disk]

        res = self.vla_for(disk).run(hanoi_instruction(disk, to_peg), max_steps=max_steps, success=done)
        res["move"] = [disk, src, to_peg]
        res["ok"] = bool(res.get("success"))
        res["pegs"] = self.arm.get_hanoi_state()
        return res

    def confirm(self, ok: bool) -> dict:
        return {"status": "not_needed", "note": "En simulación el resultado se comprueba solo.",
                "pegs": self.arm.get_hanoi_state()}

    def set_state(self, pegs: dict) -> dict:
        return {"error": "set_state sólo se usa con el brazo real"}


class RealBoard:
    """Tablero lógico para el SO-101 real. Estado en real/hanoi_state.json."""

    kind = "real"

    def __init__(self, arm, path: Path = STATE_FILE, seconds: float = 20.0):
        self.arm = arm
        self.path = path
        self.seconds = seconds
        self._data = self._load()

    # ---------------------------------------------------------- persistencia
    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"pegs": default_pegs(3), "pending": None}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2))

    # ------------------------------------------------------------- interfaz
    def state(self) -> dict:
        pegs = self._data["pegs"]
        return {"pegs": pegs, "held": None, "disks": [c for v in pegs.values() for c in v],
                "pending": self._data["pending"],
                "note": "Estado lógico: se actualiza con confirm(). Verifícalo con las cámaras."}

    def reset(self, n: int = 3) -> dict:
        if not 1 <= n <= len(DISK_ORDER):
            return {"error": f"n debe estar entre 1 y {len(DISK_ORDER)}"}
        self._data = {"pegs": default_pegs(n), "pending": None}
        self._save()
        return self.state()

    def set_state(self, pegs: dict) -> dict:
        err = _valid_pegs(pegs)
        if err:
            return {"error": err}
        self._data["pegs"] = {p: list(pegs.get(p, [])) for p in PEGS}
        self._data["pending"] = None
        self._save()
        return self.state()

    def pick_place(self, disk: str, to_peg: str, max_steps: int = 0) -> dict:
        if self._data["pending"]:
            return {"ok": False, "error": "hay un movimiento sin confirmar; llama antes a hanoi_confirm"}
        src, err = check_move(self._data["pegs"], disk, to_peg)
        if err:
            return {"ok": False, "error": err, "pegs": self._data["pegs"]}
        res = self.arm.run_skill(hanoi_instruction(disk, to_peg), seconds=self.seconds,
                                 repo_id=self.arm.policy_for(disk))
        if res.get("status") == "blocked":
            return res
        self._data["pending"] = {"disk": disk, "from": src, "to": to_peg}
        self._save()
        images = {c: str(self.arm.render(c)) for c in ("top", "wrist", "table") if c != "table" or self.arm.has_table}
        return {"status": "needs_confirmation", "vla": res, "move": [disk, src, to_peg],
                "images": images,
                "next": "Mira las imágenes. Si el disco quedó encima de " + peg_label(to_peg)
                        + ", llama a hanoi_confirm(ok=true); si no, hanoi_confirm(ok=false) y corrige."}

    def confirm(self, ok: bool) -> dict:
        pend = self._data["pending"]
        if not pend:
            return {"error": "no hay movimiento pendiente"}
        if ok:
            disk, src, to = pend["disk"], pend["from"], pend["to"]
            pegs = self._data["pegs"]
            if pegs[src][-1:] == [disk]:
                pegs[src].pop()
                pegs[to].append(disk)
        self._data["pending"] = None
        self._save()
        return {"status": "confirmed" if ok else "rejected", **self.state()}


def peg_label(peg: str) -> str:
    from hanoi_env import PEG_COLOR_NAME

    return f"{PEG_COLOR_NAME[peg]} ({peg})"
