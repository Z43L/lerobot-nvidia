"""Crea un dataset YOLO a partir de las fotos en bruto de Camera/ con pseudo-etiquetas automáticas.

Flujo:
  1. Pasa un detector ya entrenado (por defecto vision/tablero/pretrained/yamero999_best.pt,
     12 clases white_*/black_*) por cada foto de Camera/.
  2. Guarda las cajas con confianza >= --conf como etiquetas YOLO (formato txt normalizado).
  3. Reparte en train/val (semilla fija) y escribe data.yaml.
  4. Dibuja las cajas sobre cada foto en review/ para revisarlas a ojo, y un resumen en summary.json.

Uso (desde la raíz del proyecto):
  ./.venv312/bin/python vision/build_camera_dataset.py
  ./.venv312/bin/python vision/build_camera_dataset.py --conf 0.4 --device mps

No modifica Camera/. Las pseudo-etiquetas NO están revisadas: hay que mirar review/ antes de entrenar.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "Camera"), help="carpeta con las fotos en bruto")
    p.add_argument("--model", default=str(ROOT / "runs/detect/vision/runs/pieces/weights/best.pt"),
                   help="detector entrenado con vision/datasets (13 clases de vision/data.yaml)")
    p.add_argument("--out", default=str(ROOT / "vision/dataset_camara"))
    p.add_argument("--conf", type=float, default=0.25, help="confianza mínima para aceptar una caja")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--roi", default="575,150,1125,530",
                   help="x0,y0,x1,y1 del tablero en la foto; se recorta y amplía antes de detectar. 'none' = foto entera")
    p.add_argument("--device", default="mps", help="mps (por defecto, AGENTS.md), cpu o cuda")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    src, out = Path(a.src), Path(a.out)
    images = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"No hay imágenes en {src}")

    roi = None if a.roi.lower() == "none" else tuple(int(v) for v in a.roi.split(","))
    model = YOLO(a.model)
    names = model.names  # id -> nombre de clase
    print(f"{len(images)} fotos, {len(names)} clases, device={a.device}, conf>={a.conf}")

    # Carpetas de salida (no se borra nada previo: si ya existe, se sobrescriben solo los ficheros generados).
    rnd = random.Random(a.seed)
    shuffled = images[:]
    rnd.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * a.val_frac))
    split = {p: ("val" if i < n_val else "train") for i, p in enumerate(shuffled)}
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "review"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    per_image = []
    class_counts = {n: 0 for n in names.values()}
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[skip] no se puede leer {img_path.name}")
            continue
        h, w = frame.shape[:2]
        # Recorte al tablero: las piezas son pequeñas en la foto completa y el detector va mucho mejor ampliado.
        ox, oy, src_img = 0, 0, frame
        if roi is not None:
            ox, oy = roi[0], roi[1]
            src_img = frame[roi[1]:roi[3], roi[0]:roi[2]]
        res = model.predict(
            source=src_img, conf=a.conf, iou=a.iou, imgsz=a.imgsz, device=a.device,
            agnostic_nms=True, verbose=False  # una sola caja por pieza, aunque la clase dude
        )[0]

        lines, review = [], frame.copy()
        for box in res.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, x2, y1, y2 = x1 + ox, x2 + ox, y1 + oy, y2 + oy  # a coordenadas de la foto completa
            lines.append(
                f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
            class_counts[names[cls]] += 1
            cv2.rectangle(review, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(review, f"{names[cls]} {float(box.conf[0]):.2f}", (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        stem = img_path.stem
        sp = split[img_path]
        # Copia la imagen al split; etiquetas en .txt (vacío si no hay detecciones, que cuenta como fondo).
        cv2.imwrite(str(out / "images" / sp / f"{stem}.jpg"), frame)
        (out / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        cv2.imwrite(str(out / "review" / f"{stem}.jpg"), review)
        per_image.append({"file": img_path.name, "split": sp, "boxes": len(lines)})
        print(f"[{sp:5}] {img_path.name}: {len(lines)} cajas")

    data_yaml = out / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in names.items())
    data_yaml.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"names:\n{names_block}\n"
    )

    summary = {
        "source": str(src.resolve()),
        "model": str(Path(a.model).resolve()),
        "conf": a.conf,
        "iou": a.iou,
        "imgsz": a.imgsz,
        "images": len(per_image),
        "train": sum(1 for r in per_image if r["split"] == "train"),
        "val": sum(1 for r in per_image if r["split"] == "val"),
        "boxes_total": sum(r["boxes"] for r in per_image),
        "boxes_per_class": class_counts,
        "images_without_boxes": [r["file"] for r in per_image if r["boxes"] == 0],
        "per_image": per_image,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n--- Resumen ---")
    print(f"fotos: {summary['images']} (train {summary['train']}, val {summary['val']})")
    print(f"cajas totales: {summary['boxes_total']}")
    for n, c in class_counts.items():
        print(f"  {n:14} {c}")
    print(f"sin cajas: {len(summary['images_without_boxes'])}")
    print(f"data.yaml: {data_yaml}")
    print(f"revisión visual: {out / 'review'}")


if __name__ == "__main__":
    main()
