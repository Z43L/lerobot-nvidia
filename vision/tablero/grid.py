import numpy as np
TL, TR, BR, BL = [(607,186),(997,172),(1098,495),(608,492)]
def pt(u, v):
    top = np.array(TL, float) + (np.array(TR, float)-np.array(TL, float))*u/8
    bot = np.array(BL, float) + (np.array(BR, float)-np.array(BL, float))*u/8
    return top + (bot-top)*v/8
