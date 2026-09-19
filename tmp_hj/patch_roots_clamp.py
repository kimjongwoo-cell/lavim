import hashlib, os, shutil, sys
P = "/home/users/whddn12316/wsi_latent_0915_decode_hj/vision_text_mas/onepass_navigation_roots.py"
src = open(P).read()
if not hashlib.md5(src.encode()).hexdigest().startswith("a36830f5"):
    sys.exit("roots md5 != a36830f5; abort")
old = "        boxes = edge_anchored_grid(root, side=EXACT_X5_SIDE, rows=4, cols=6)\n"
new = ("        # A root cell is >= EXACT_X5_SIDE on both sides unless the slide itself is narrower (PANDA: 11/196\n"
       "        # slides with a 2816-3840 px side), where a fixed 4096 px box cannot be placed. Clamp the x5 side to\n"
       "        # the root so those slides get candidates; every other slide keeps side == EXACT_X5_SIDE (09-15).\n"
       "        side = min(EXACT_X5_SIDE, int(root.width), int(root.height))\n"
       "        boxes = edge_anchored_grid(root, side=side, rows=4, cols=6)\n")
assert src.count(old) == 1
out = src.replace(old, new)
compile(out, P, "exec")
shutil.copy2(P, P + ".bak_0915_x5clamp")
tmp = P + ".tmp_x5clamp"
open(tmp, "w").write(out)
os.replace(tmp, P)
print("patched", hashlib.md5(out.encode()).hexdigest()[:8])
