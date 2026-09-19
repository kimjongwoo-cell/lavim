import numpy as np
P, M, T = 16, 2, 2

def crops_px(pv, thw):
    out, off = [], 0
    for t, h, w in thw:
        n = t * h * w
        x = pv[off:off + n].reshape(n, 3, T, P, P)[:, :, 0]
        gh, gw = h // M, w // M
        x = x.reshape(gh, gw, M, M, 3, P, P).transpose(0, 2, 5, 1, 3, 6, 4).reshape(gh * M * P, gw * M * P, 3)
        out.append((np.clip((x * 0.5 + 0.5) * 255, 0, 255), gh, gw))
        off += n
    return out


