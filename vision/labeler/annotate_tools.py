"""Herramientas para anotar el dataset de piezas viendo las fotos (candidatas numeradas -> clases -> YOLO).

  candidates        detecta cajas candidatas (conf baja) en cada foto -> labeler/candidates.json
  render N [N...]   recorte 2x de la foto N con la posición ANTERIOR superpuesta (casillas y cajas) -> labeler/crops/NN.png
  apply             lee labeler/annotations.txt y escribe etiquetas YOLO + posiciones + overlays

Formato de annotations.txt: una línea por foto, la posición se HEREDA de la foto anterior y se edita con:
  NN: pos wKe1 wQd1 bPa7 ...   posición completa (vacía el tablero antes). Pieza = color(w/b)+tipo(KQRBNP)+casilla
  NN: e2e4                     mueve la pieza de e2 a e4 (si e4 tiene pieza, la sustituye)
  NN: xd5                      quita la pieza de d5 (captura / pieza fuera del tablero)
  NN: +wPa2                    añade una pieza
  NN: @e4=x1,y1,x2,y2          fuerza la caja de la pieza de e4 (píxeles del recorte 2x, los de la regla)
  NN: =                        misma posición que la anterior
Se pueden combinar varios tokens en una línea, se aplican en orden.
Orientación asumida (comprobada con rey en e1/dama en d1): blancas a la derecha de la foto, a1 = esquina inferior-derecha.
"""
import json, os, sys, re
from pathlib import Path
import cv2, numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DS = ROOT / "vision/dataset_camara"
MODEL = ROOT / "runs/detect/vision/runs/pieces/weights/best.pt"
# Esquinas del área de juego 8x8 (medidas en la foto 72 con el tablero vacío), TL TR BR BL en px de la foto.
CORNERS = np.array([[632.5,197.5],[975,195],[1055,470],[657.5,472.5]], float)
X0, Y0, X1, Y1 = 560, 140, 1140, 545
SC = 2.0
W, H = 1280, 720
CODES = {f"{c}{p}": f"{'white' if c=='w' else 'black'}-{n}" for c in "wb"
         for p, n in zip("pnbrqk", ["pawn","knight","bishop","rook","queen","king"])}
FILES = "abcdefgh"

def pt(u, v):
    TL, TR, BR, BL = CORNERS
    top = TL + (TR - TL) * u / 8
    bot = BL + (BR - BL) * u / 8
    return top + (bot - top) * v / 8

def square_of(x, y):
    """Casilla (p. ej. 'e4') del punto (x,y) de la foto, o None si cae fuera de la rejilla."""
    src = CORNERS.astype(np.float32)
    dst = np.array([[0,0],[8,0],[8,8],[0,8]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    u, v = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), M)[0][0]
    if not (0 <= u < 8 and 0 <= v < 8):
        return None
    return f"{FILES[7 - int(v)]}{8 - int(u)}"

def entries():
    return json.loads((DS / "summary.json").read_text())["per_image"]

def image_path(rec):
    return DS / "images" / rec["split"] / (Path(rec["file"]).stem + ".jpg")

def class_ids():
    names = {}
    in_b = False
    for line in (DS / "data.yaml").read_text().splitlines():
        if line.startswith("names:"): in_b = True; continue
        m = re.match(r"\s+(\d+):\s*(.+)$", line)
        if in_b and m: names[m.group(2).strip()] = int(m.group(1))
    return names

def cmd_candidates():
    from ultralytics import YOLO
    model = YOLO(str(MODEL))
    out = []
    for n, rec in enumerate(entries(), 1):
        img = cv2.imread(str(image_path(rec)))
        r = model.predict(img[Y0:Y1, X0:X1], conf=0.12, iou=0.45, imgsz=960, device="mps",
                          agnostic_nms=True, verbose=False)[0]
        boxes = [[float(v) + o for v, o in zip(b.xyxy[0].tolist(), (X0, Y0, X0, Y0))] for b in r.boxes]
        boxes.sort(key=lambda b: (b[1] + b[3]) / 2)
        out.append({"n": n, "file": rec["file"], "boxes": boxes})
    (HERE / "candidates.json").write_text(json.dumps(out))
    print("candidatas:", sum(len(o["boxes"]) for o in out))


# (nombre, alto, ancho) en unidades de casilla local (raíz del área proyectada); ajustado con detecciones de conf>=0.5.
PIECES = {"P": ("pawn", 1.33, 0.80), "N": ("knight", 1.60, 0.95), "B": ("bishop", 1.80, 0.85),
          "R": ("rook", 1.55, 0.90), "Q": ("queen", 2.05, 1.00), "K": ("king", 2.25, 1.00)}
_M = cv2.getPerspectiveTransform(CORNERS.astype(np.float32), np.array([[0,0],[8,0],[8,8],[0,8]], np.float32))
_Mi = np.linalg.inv(_M)

def to_xy(u, v):
    return cv2.perspectiveTransform(np.array([[[u, v]]], np.float32), _Mi)[0][0]

def sq_uv(sq):
    """Centro de la casilla en coordenadas de rejilla (u: rango, v: archivo)."""
    return 7.5 - (int(sq[1]) - 1), 7.5 - FILES.index(sq[0])

def synth_box(piece, sq):
    """Caja sintética (x1,y1,x2,y2 en px de la foto) de una pieza sobre una casilla, por proporciones calibradas."""
    _, hh, ww = PIECES[piece[1]]
    u, v = sq_uv(sq)
    c = to_xy(u, v)
    q = [to_xy(u + du, v + dv) for du, dv in [(-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)]]
    area = 0.5 * abs(sum(q[i][0]*q[(i+1)%4][1] - q[(i+1)%4][0]*q[i][1] for i in range(4)))
    sc = area ** .5
    # desplazamiento de la caja respecto al centro de la casilla (la perspectiva inclina las piezas): regresión lineal
    gx, gy = (c[0] - 840) / 100, (c[1] - 330) / 100
    cx = c[0] + (0.052 + 0.104 * gx + 0.049 * gy) * sc
    y2 = c[1] + (0.265 - 0.060 * gx + 0.033 * gy) * sc
    return [cx - ww * sc / 2, y2 - hh * sc, cx + ww * sc / 2, y2]

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)

TOK = re.compile(r"^(pos|=|[a-h][1-8][a-h][1-8]|x[a-h][1-8]|\+[wb][KQRBNP][a-h][1-8]|@[a-h][1-8]=[\d.,]+|[wb][KQRBNP][a-h][1-8])$")

def parse_annotations():
    """-> {n: [tokens]} (n 1-based)."""
    ann = {}
    for line in (HERE / "annotations.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line: continue
        m = re.match(r"^(\d+):\s*(.*)$", line)
        if not m: raise SystemExit(f"línea mal formada: {line}")
        toks = m.group(2).split()
        for t in toks:
            if not TOK.match(t): raise SystemExit(f"foto {m.group(1)}: token no válido '{t}'")
        ann[int(m.group(1))] = toks
    return ann

def step(state, overrides, toks):
    """Aplica los tokens a state {casilla: 'wP'} y overrides {casilla: caja 2x}. Devuelve nuevo state."""
    state = dict(state)
    for t in toks:
        if t == "=":
            continue
        if t == "pos":
            state = {}; overrides.clear()
        elif t.startswith("+"):
            state[t[3:]] = t[1:3]
        elif t.startswith("x"):
            state.pop(t[1:], None); overrides.pop(t[1:], None)
        elif t.startswith("@"):
            sq, vals = t[1:].split("=")
            overrides[sq] = [float(v) for v in vals.split(",")]
        elif re.match(r"^[a-h][1-8][a-h][1-8]$", t):
            src, dst = t[:2], t[2:]
            if src not in state: raise SystemExit(f"movimiento {t}: no hay pieza en {src}")
            state[dst] = state.pop(src)
            overrides.pop(src, None); overrides.pop(dst, None)
        else:  # wKe1
            state[t[2:]] = t[:2]
    return state

def build_boxes(state, overrides, cands):
    """Cajas finales {casilla: (pieza, caja px foto, origen)}; se ajusta a una candidata del detector si coincide bien."""
    out = {}
    for sq, piece in state.items():
        if sq in overrides:
            a, b, c, d = overrides[sq]
            out[sq] = (piece, [a / SC + X0, b / SC + Y0, c / SC + X0, d / SC + Y0], "manual")
            continue
        sb = synth_box(piece, sq)
        best = max(cands, key=lambda cb: iou(sb, cb), default=None)
        if best is not None and iou(sb, best) > 0.5:
            out[sq] = (piece, list(best), "detector")
        else:
            out[sq] = (piece, sb, "sintética")
    return out

def to_fen(state):
    """Colocación de piezas en notación FEN (solo el campo de piezas) a partir de {casilla: 'wK'}."""
    rows = []
    for rank in range(8, 0, -1):
        row, empty = "", 0
        for f in FILES:
            p = state.get(f"{f}{rank}")
            if p is None:
                empty += 1
            else:
                if empty: row += str(empty); empty = 0
                row += p[1].upper() if p[0] == "w" else p[1].lower()
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows)

def run_states(upto):
    """Estados tras anotar fotos 1..upto (hereda de la anterior). Devuelve lista de (state, overrides) por foto."""
    ann = parse_annotations()
    state, overrides, res = {}, {}, []
    for n in range(1, upto + 1):
        if n in ann:
            state = step(state, overrides, ann[n])
        res.append((dict(state), dict(overrides), n in ann))
    return res

def draw_overlay(img, boxes, scale=1.0, off=(0, 0), thick=1):
    for sq, (piece, (x1, y1, x2, y2), origin) in boxes.items():
        col = {"detector": (0, 220, 0), "sintética": (0, 200, 255), "manual": (255, 120, 0)}[origin]
        p1 = (int((x1 - off[0]) * scale), int((y1 - off[1]) * scale)); p2 = (int((x2 - off[0]) * scale), int((y2 - off[1]) * scale))
        cv2.rectangle(img, p1, p2, col, thick)
        cv2.putText(img, f"{piece}{sq}", (p1[0], max(10, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

def square_labels(c):
    for f in range(8):
        for r in range(8):
            u, v = sq_uv(f"{FILES[f]}{r+1}")
            x, y = to_xy(u, v)
            cv2.putText(c, f"{FILES[f]}{r+1}", (int((x - X0) * SC) - 9, int((y - Y0) * SC) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 0), 1, cv2.LINE_AA)

def cmd_render(nums):
    cand = json.loads((HERE / "candidates.json").read_text())
    (HERE / "crops").mkdir(exist_ok=True)
    ann = parse_annotations()
    for n in nums:
        rec = entries()[n - 1]
        img = cv2.imread(str(image_path(rec)))
        c = cv2.resize(img[Y0:Y1, X0:X1], None, fx=SC, fy=SC, interpolation=cv2.INTER_CUBIC)
        square_labels(c)
        # posición heredada (estado tras la foto anterior ya anotada, o tras esta si ya está anotada)
        last = max([k for k in ann if k <= n - (0 if n in ann else 1)], default=0)
        if last:
            state, ov, _ = run_states(last)[-1]
            boxes = build_boxes(state, ov, cand[n - 1]["boxes"])
            draw_overlay(c, boxes, SC, (X0, Y0))
        cv2.imwrite(str(HERE / "crops" / f"{n:02d}.png"), c)

def cmd_apply():
    cand = json.loads((HERE / "candidates.json").read_text())
    ann = parse_annotations()
    ids = class_ids()
    last = max(ann, default=0)
    states = run_states(last)
    (DS / "positions").mkdir(exist_ok=True)
    (DS / "review_final").mkdir(exist_ok=True)
    per_class, total, origins = {}, 0, {}
    for n, rec in enumerate(entries(), 1):
        if n > last: break
        state, ov, _ = states[n - 1]
        boxes = build_boxes(state, ov, cand[n - 1]["boxes"])
        lines, review = [], cv2.imread(str(image_path(rec)))
        for sq, (piece, (x1, y1, x2, y2), origin) in boxes.items():
            name = f"{'white' if piece[0]=='w' else 'black'}-{PIECES[piece[1]][0]}"
            x1, x2 = sorted((max(0, x1), min(W, x2))); y1, y2 = sorted((max(0, y1), min(H, y2)))
            lines.append(f"{ids[name]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            per_class[name] = per_class.get(name, 0) + 1
            origins[origin] = origins.get(origin, 0) + 1
        stem = Path(rec["file"]).stem
        (DS / "labels" / rec["split"] / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        (DS / "positions" / f"{stem}.json").write_text(json.dumps(
            {"fen": to_fen(state), "pieces": {sq: p for sq, p in sorted(state.items())}, "count": len(state),
             "orientation": "blancas a la derecha de la foto, a1 abajo-derecha"}, indent=1, ensure_ascii=False))
        draw_overlay(review, boxes)
        cv2.imwrite(str(DS / "review_final" / f"{n:02d}.jpg"), review[Y0:Y1, X0:X1])
        total += len(lines)
    print(f"fotos procesadas: {last}/{len(entries())}, cajas: {total}, origen de cajas: {origins}")
    for k, v in sorted(per_class.items()): print(f"  {k:14} {v}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "candidates": cmd_candidates()
    elif cmd == "render": cmd_render([int(a) for a in sys.argv[2:]])
    elif cmd == "apply": cmd_apply()
    else: print(__doc__)
