import cv2, numpy as np, sys, glob, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import pt
X0, Y0, X1, Y1 = 575, 150, 1125, 530
SC = 2.0
def render(path, out):
    im = cv2.imread(path)
    c = cv2.resize(im[Y0:Y1, X0:X1], None, fx=SC, fy=SC, interpolation=cv2.INTER_CUBIC)
    P = lambda p: (int((p[0]-X0)*SC), int((p[1]-Y0)*SC))
    for i in range(9):
        cv2.line(c, P(pt(i,0)), P(pt(i,8)), (0,255,0), 1)
        cv2.line(c, P(pt(0,i)), P(pt(8,i)), (0,255,0), 1)
    for u in range(8):
        for v in range(8):
            q = P(pt(u+0.08, v+0.1))
            cv2.putText(c, f"{u}{v}", q, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(out, c)
if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
