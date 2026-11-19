"""Genera una skill de Claude a partir de un movimiento de hub/motions/<nombre>.json.

  ./.venv312/bin/python hub/skill_gen.py <nombre>            # crea .claude/skills/motion-<nombre>/SKILL.md
  ./.venv312/bin/python hub/skill_gen.py <nombre> --force    # la regenera (sobrescribe)

La skill generada explica cómo reproducir el movimiento en simulación y en el brazo real con los
gates de so101-motion. Lleva una cabecera "generada": no se edita a mano; se regenera desde el JSON.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parent
ROOT = HUB.parent
sys.path.insert(0, str(HUB))

import motion_lib  # noqa: E402
import trajectory as T  # noqa: E402

SKILLS = ROOT / ".claude" / "skills"
MARK = "<!-- generada por hub/skill_gen.py: no editar a mano, se regenera desde hub/motions -->"


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not s:
        raise ValueError("Nombre de movimiento no válido.")
    return s


def render(motion: dict) -> str:
    name = motion["name"]
    summary = T.playback_summary(motion)
    kfs = motion["keyframes"]
    desc = motion.get("description") or f"Movimiento grabado: {name}"
    rows = "\n".join(
        f"| {i} | {kf.get('t', 0):.2f} | " + " | ".join(f"{v:.1f}" for v in kf["deg"]) + f" | {kf.get('note', '')} |"
        for i, kf in enumerate(kfs)
    )
    return f"""---
name: motion-{slug(name)}
description: {desc}. Reproduce el movimiento "{name}" ({summary['keyframes']} keyframes, ~{summary['seconds']} s) en la simulación y en el SO-101 real con los gates de so101-motion. Úsala cuando el usuario pida "haz {name}", "repite {name}" o "pásalo al brazo".
---

{MARK}

# Movimiento: {name}

{desc}

Duración de reproducción a {int(T.max_deg_per_tick())}° por paso: **{summary['seconds']} s** ({summary['ticks']} pasos a 30 Hz).
Paso máximo registrado: {summary['max_step_deg_per_tick']}° por paso.

## Keyframes (grados; pinza 0..100)

| # | t (s) | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | nota |
|---|---|---|---|---|---|---|---|---|
{rows}

## Simulación

```bash
./.venv312/bin/python hub/motion_lib.py validate {name}
./.venv312/bin/python hub/motion_lib.py sim {name}
```

Para verlo en la ventana: mueve el brazo de `so101-sim-objects` por keyframes con `move_joints` (en radianes, usando `hub/motion_lib.py` para convertir) y comprueba que la ventana `objects_env.py --follow` lo sigue.

## Brazo real

Reproducción con una sola confirmación para toda la trayectoria. Solo con el sí explícito del usuario, con el área libre y una persona junto al brazo.

1. `so101-motion > motion_status`. Si `allow_real_motion` es false, lo activa el usuario; tú no.
2. Valida con la calibración real: `hub/motion_lib.py validate {name}`. Si falla, no sigas.
3. `trajectory_preview("{name}")`: enseña al usuario el plan (duración, pose inicial, pasos máximos, rangos) y pide un único sí.
4. Con el sí: `trajectory_execute("{name}", confirm_token)`. Va a 30 Hz y aborta si el brazo se queda atrás más de 15° durante 5 pasos.
5. Si algo se mueve raro en cualquier momento: `motion_stop` de inmediato.

## Reglas

- No cambies los valores de la tabla a mano: vuelve a grabar o edita el JSON y regenera.
- La pinza va en 0..100, igual en sim y real.
- Lo que no se ha probado en el brazo real se marca como pendiente.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    motion = motion_lib.load(args.name)
    problems = motion_lib.validate(motion)
    if problems:
        print("No se genera la skill, el movimiento tiene problemas:")
        for p in problems:
            print(" -", p)
        return 1
    out_dir = SKILLS / f"motion-{slug(args.name)}"
    out = out_dir / "SKILL.md"
    if out.exists() and not args.force and MARK not in out.read_text():
        print(f"{out} existe y no es una skill generada. Usa otro nombre o --force.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(render(motion))
    print(f"Skill creada: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
