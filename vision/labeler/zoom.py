"""Zoom con rejilla y nombres de casilla sobre una foto: zoom.py N X0 Y0 X1 Y1 [Z] [salida.png]  (X,Y en px de la foto)"""
import cv2, sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import annotate_tools as T
n = int(sys.argv[1]); X0, Y0, X1, Y1 = map(int, sys.argv[2:6]); Z = float(sys.argv[6]) if len(sys.argv) > 6 else 3.5
out = sys.argv[7] if len(sys.argv) > 7 else "/tmp/zoom.png"
im = cv2.imread(str(T.image_path(T.entries()[n - 1])))
c = cv2.resize(im[Y0:Y1, X0:X1], None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC)
P = lambda p: (int((p[0] - X0) * Z), int((p[1] - Y0) * Z))
for i in range(9):
    cv2.line(c, P(T.to_xy(i, 0)), P(T.to_xy(i, 8)), (0, 255, 0), 1)
    cv2.line(c, P(T.to_xy(0, i)), P(T.to_xy(8, i)), (0, 255, 0), 1)
for f in range(8):
    for r in range(8):
        x, y = T.to_xy(*T.sq_uv(f"{T.FILES[f]}{r+1}"))
        if X0 < x < X1 and Y0 < y < Y1:
            cv2.putText(c, f"{T.FILES[f]}{r+1}", (P((x, y))[0] - 10, P((x, y))[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
cv2.putText(c, str(n), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imwrite(out, c)
