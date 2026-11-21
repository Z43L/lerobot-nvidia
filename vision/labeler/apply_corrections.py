"""Vuelca las correcciones exportadas por editor.html a las etiquetas YOLO de vision/dataset_camara/.

  ./.venv312/bin/python vision/labeler/apply_corrections.py ~/Downloads/corrections.json

Antes de escribir, guarda una copia de las etiquetas actuales en vision/dataset_camara/labels_auto_backup/
(solo la primera vez). No toca las imágenes.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "dataset_camara"
LABELS = DATASET / "labels"
BACKUP = DATASET / "labels_auto_backup"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Uso: apply_corrections.py <corrections.json>")
    data = json.loads(Path(sys.argv[1]).read_text())
    if data.get("format") != "yolo-corrections-v1":
        raise SystemExit("El fichero no es una exportación del editor (format distinto).")

    if not BACKUP.exists():
        shutil.copytree(LABELS, BACKUP)
        print(f"copia de seguridad de las etiquetas actuales -> {BACKUP}")

    n_files = n_boxes = 0
    for img in data["images"]:
        out = LABELS / img["split"] / f"{img['file']}.txt"
        lines = [
            f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
            for b in img["boxes"]
        ]
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_files += 1
        n_boxes += len(lines)

    names = data.get("classes", {})
    print(f"{n_files} etiquetas escritas, {n_boxes} cajas en total.")
    print(f"clases: {', '.join(f'{k}={v}' for k, v in names.items())}")


if __name__ == "__main__":
    main()
