import numpy as np
from scipy import ndimage as ndi
def cell_labels(rgb512, cell=32):
    """평가용 픽셀 정답 (토큰 셀 단위). 흰 셀(rho>=0.6) 중
    막(membrane) 픽셀: 연한 분홍/회색 선 = gray<228 또는 채도>0.07 인 얇은 구조.
    glass: 막 비율<0.01, fat: 막 비율 0.02~0.35. 나머지는 -1(애매)."""
    x = rgb512.astype(np.float32) / 255
    gray = (x @ np.array([0.299, 0.587, 0.114])) * 255
    mx, mn = x.max(-1), x.min(-1); sat = (mx - mn) / np.maximum(mx, 1e-6)
    white = gray > 215
    memb = (gray < 228) | (sat > 0.07)
    H, W = gray.shape; gh, gw = H // cell, W // cell
    rho = white.reshape(gh, cell, gw, cell).mean((1, 3))
    mf = memb.reshape(gh, cell, gw, cell).mean((1, 3))
    lab = -np.ones((gh, gw), int)
    wc = rho >= 0.6
    lab[wc & (mf < 0.01)] = 0          # glass
    lab[wc & (mf >= 0.02) & (mf <= 0.35)] = 1   # fat (막 있는 흰 셀)
    return rho.reshape(-1), mf.reshape(-1), lab.reshape(-1)
