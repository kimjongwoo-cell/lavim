import sys
p=sys.argv[1]; s=open(p).read()
a='''        elif mode in ("wsi", "wsi_flat", "wsi_shuf"):'''
n='''        elif mode in ("wsi", "wsi_flat", "wsi_shuf", "scale_v2", "scale_v2_flat", "scale_v2_shuf"):'''
assert s.count(a)==1 and "scale_v2" not in s; s=s.replace(a,n)
a2='''                constraint={"wsi": "hier", "wsi_flat": "flat",
                            "wsi_shuf": "shuffled"}[mode],
                return_info=True)'''
n2='''                constraint={"wsi": "hier", "wsi_flat": "flat", "wsi_shuf": "shuffled",
                            "scale_v2": "hier", "scale_v2_flat": "flat",
                            "scale_v2_shuf": "shuffled"}[mode],
                return_info=True,
                # scale_v2 (0911 spec): identical rule to wsi but the coverage
                # radius uses raw ||u_j - u_k|| (no per-grid diam scaling)
                normalize_diam=not mode.startswith("scale_v2"))'''
assert s.count(a2)==1; s=s.replace(a2,n2)
open(p,"w").write(s); print("patched",p)
