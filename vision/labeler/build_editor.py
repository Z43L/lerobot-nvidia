"""Genera vision/labeler/editor.html: editor local para corregir a mano las etiquetas de vision/dataset_camara/.

  ./.venv312/bin/python vision/labeler/build_editor.py

Luego abre vision/labeler/editor.html en el navegador (doble clic; no hace falta servidor).
Al terminar, pulsa "Exportar corrección" y guarda el JSON. Después:

  ./.venv312/bin/python vision/labeler/apply_corrections.py <ruta del JSON exportado>

Este script no modifica las etiquetas; solo crea el HTML con los datos actuales.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "dataset_camara"
TEMPLATE = HERE / "editor_template.html"
OUT = HERE / "editor.html"


def read_names() -> dict[int, str]:
    names = {}
    in_block = False
    for line in (DATASET / "data.yaml").read_text().splitlines():
        if line.startswith("names:"):
            in_block = True
            continue
        if in_block:
            m = re.match(r"\s+(\d+):\s*(.+)$", line)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
    return names


def read_boxes(label_file: Path) -> list[dict]:
    boxes = []
    if not label_file.exists():
        return boxes
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        boxes.append({"cls": c, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def main() -> None:
    names = read_names()
    summary = json.loads((DATASET / "summary.json").read_text())
    images = []
    for rec in summary["per_image"]:
        split, file = rec["split"], rec["file"]
        stem = Path(file).stem
        src = f"../dataset_camara/images/{split}/{stem}.jpg"
        images.append({
            "file": stem,
            "split": split,
            "src": quote(src),
            "boxes": read_boxes(DATASET / "labels" / split / f"{stem}.txt"),
        })
    data = {"names": names, "images": images}
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(data, ensure_ascii=False))
    OUT.write_text(html)
    print(f"{len(images)} imágenes, {sum(len(i['boxes']) for i in images)} cajas -> {OUT}")


if __name__ == "__main__":
    main()
