"""Add a demand axis to vit_146clamp_gtex.py: page / slide / unif_g / unif_n x tau."""
from pathlib import Path

p = Path("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/vit_146clamp_gtex.py")
s = p.read_text()

# --- null template b, needed to rebuild the unnormalised residual r_g = exp(x_g - gamma*b)
s = s.replace(
    'BG_TISSUE = 0.10',
    '''NULL = Path("/home/users/whddn12316/wsi_latent_0902_2155_hj/sender_relay_exp/runs/vit_nullcal/null_maps.npz")
BLOCK = 23
EPS = 1e-3
BG_TISSUE = 0.10''', 1)

s = s.replace('# --------------------------------------------------------------------------- pure',
'''# --------------------------------------------------------------------------- pure


def centred_log(maps, eps: float = EPS) -> np.ndarray:
    lg = np.log(np.asarray(maps, dtype=float) + eps)
    return lg - lg.mean(axis=-1, keepdims=True)


def demands(a_raw: np.ndarray, gamma: float, b: np.ndarray, counts) -> dict:
    """The four demand weights of the #14.6 audit, rebuilt from the saved a_raw and gamma.

    page   w_gi = softmax_g(x_gi - gamma b_i)        support-local salience   (= V0, stored w_page)
    slide  w_i  = exp(x_i - gamma b) / sum over ALL  global salience          (= V3)
    unif_g w_gi = 1 / N_g                            support-local uniform    (= V2)
    unif_n w_i  = 1 / N                              global uniform
    """
    counts = [int(c) for c in counts]
    offs = np.cumsum([0] + counts)
    N = int(offs[-1])
    resid = []
    for g in range(len(counts)):
        x = centred_log(a_raw[offs[g]:offs[g + 1]])
        resid.append(np.exp(x - gamma * b))
    tot = sum(float(r.sum()) for r in resid)
    return {
        "page": np.concatenate([r / r.sum() for r in resid]),
        "slide": np.concatenate(resid) / tot,
        "unif_g": np.concatenate([np.full(c, 1.0 / c) for c in counts]),
        "unif_n": np.full(N, 1.0 / N),
    }''', 1)

# --- CLI
s = s.replace('ap.add_argument("--taus", default="2,1,0.5,0.25")',
              'ap.add_argument("--taus", default="2,1,0.5,0.25")\n'
              '    ap.add_argument("--demands", default="page,slide,unif_g,unif_n")', 1)
s = s.replace('    taus = [float(t) for t in args.taus.split(",")]',
              '    taus = [float(t) for t in args.taus.split(",")]\n'
              '    dems = [x.strip() for x in args.demands.split(",")]\n'
              '    b = centred_log(np.load(NULL)["maps"][:, BLOCK]).mean(axis=0)', 1)

# --- per case: build all demands, check the page one reproduces the stored w_page
s = s.replace('        rec = {"N": N, "k": k, "gamma": float(d["gamma"]), "taus": {}}',
'''        W = demands(d["a_raw"].astype(np.float64), float(d["gamma"]), b, counts)
        chk = float(np.abs(W["page"] - w_page).max())
        rec = {"N": N, "k": k, "gamma": float(d["gamma"]), "w_page_maxdiff": chk, "taus": {}}
        assert chk < 1e-9, f"w_page rebuild mismatch {chk}"''', 1)

# --- loop over demand x tau
s = s.replace('''        for tau in taus:
            order, gains, F, Fmax = weighted_greedy(z, w_page, counts, k, tau)''',
'''        for dem in dems:
          for tau in taus:
            w = W[dem]
            order, gains, F, Fmax = weighted_greedy(z, w, counts, k, tau)''', 1)
s = s.replace('            rec["taus"][str(tau)] = {',
              '            rec["taus"][f"{dem}|{tau}"] = {', 1)
s = s.replace('''                "first_share": first_pick_share(z, w_page, counts, k, tau),''',
              '''                "first_share": first_pick_share(z, w, counts, k, tau),''', 1)
s = s.replace('''            agg[tau].append(rec["taus"][str(tau)])''',
              '''            agg[f"{dem}|{tau}"].append(rec["taus"][f"{dem}|{tau}"])''', 1)
s = s.replace('    per_case, agg = {}, {t: [] for t in taus}',
              '    per_case, agg = {}, {f"{dm}|{t}": [] for dm in dems for t in taus}', 1)

# --- per-case print
s = s.replace('''        r0 = rec["taus"][str(taus[0])]
        r1 = rec["taus"][str(taus[1])] if len(taus) > 1 else r0''',
'''        r0 = rec["taus"][f"{dems[0]}|{taus[0]}"]
        r1 = rec["taus"][f"{dems[min(1, len(dems) - 1)]}|{taus[min(1, len(taus) - 1)]}"]''', 1)
s = s.replace('''              f"tau={taus[0]} sd {r0['sd']:.1f} first {r0['first_share']:.3f} | "
              f"tau={taus[1] if len(taus) > 1 else taus[0]} sd {r1['sd']:.1f} "''',
'''              f"{dems[0]}|{taus[0]} sd {r0['sd']:.1f} first {r0['first_share']:.3f} | "
              f"{dems[min(1, len(dems) - 1)]}|{taus[min(1, len(taus) - 1)]} sd {r1['sd']:.1f} "''', 1)

# --- aggregation + final table
s = s.replace('''    names = list(next(iter(per_case.values()))["taus"][str(taus[0])]["overlap"])
    for tau in taus:
        rows = agg[tau]
        summary["mean"][str(tau)] = {''',
'''    keys = [f"{dm}|{t}" for dm in dems for t in taus]
    names = list(next(iter(per_case.values()))["taus"][keys[0]]["overlap"])
    for kk in keys:
        rows = agg[kk]
        summary["mean"][kk] = {''', 1)
s = s.replace('''        summary["mean"][str(tau)]["overlap"] = {n: mean([r["overlap"][n] for r in rows])
                                                for n in names}''',
'''        summary["mean"][kk]["overlap"] = {n: mean([r["overlap"][n] for r in rows]) for n in names}''', 1)
s = s.replace('''    print(f"{'tau':>6} {'cos cutoff':>11} {'crop sd':>8} {'min':>5} {'max':>5} "
          f"{'first-pick share':>17} {'coverage':>9} {'sel tissue':>11} {'sel bg':>7} "
          f"{'hot8':>6} {'ov #14.6':>9}")
    for tau in taus:
        m = summary["mean"][str(tau)]
        print(f"{tau:>6} {1 - tau / 2:>11.2f} {m['sd']:>8.1f} {m['min']:>5.0f} {m['max']:>5.0f} "
              f"{m['first_share']:>17.3f} {m['coverage_frac']:>9.3f} {m['sel_tissue']:>11.3f} "
              f"{m['sel_bg']:>7.3f} {m['hot8_share']:>6.3f} "
              f"{m['overlap']['#14.6 residual + support coverage']:>9.3f}")''',
'''    print(f"{'demand':>7} {'tau':>5} {'cutoff':>7} {'crop sd':>8} {'min':>5} {'max':>5} "
          f"{'first':>6} {'cover':>6} {'tissue':>7} {'bg':>6} {'hot8':>6} "
          f"{'ov146':>6} {'ov topk':>8}")
    for dm in dems:
        for tau in taus:
            m = summary["mean"][f"{dm}|{tau}"]
            print(f"{dm:>7} {tau:>5} {1 - tau / 2:>7.2f} {m['sd']:>8.1f} {m['min']:>5.0f} "
                  f"{m['max']:>5.0f} {m['first_share']:>6.3f} {m['coverage_frac']:>6.3f} "
                  f"{m['sel_tissue']:>7.3f} {m['sel_bg']:>6.3f} {m['hot8_share']:>6.3f} "
                  f"{m['overlap']['#14.6 residual + support coverage']:>6.3f} "
                  f"{m['overlap']['#14.6 residual top-k']:>8.3f}")''', 1)
s = s.replace('''    m0 = summary["mean"][str(taus[0])]''', '''    m0 = summary["mean"][keys[0]]''', 1)
s = s.replace('''    for n in names:
        row = " ".join(f"{summary['mean'][str(t)]['overlap'][n]:.3f}" for t in taus)
        print(f"     tau {taus} -> {row}   {n}")''',
'''    for n in names:
        row = " ".join(f"{summary['mean'][kk]['overlap'][n]:.3f}" for kk in keys)
        print(f"     {row}   {n}")
    print(f"     (columns: {' '.join(keys)})")''', 1)

p_out = Path("/tmp/patch_out.py")
print(s.count('for dem in dems'), s.count('def demands'), len(s))
p.write_text(s)
print("patched", p)
