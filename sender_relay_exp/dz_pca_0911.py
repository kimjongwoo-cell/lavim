import json, re, sys, numpy as np
K="/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/runs/vcut"
DS=json.load(open("/home/users/whddn12316/wsi_latent_0915_decode_hj/sender_relay_exp/smoke/gtex40.json"))
def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def rows(arm):
    out={}
    for l in open(f"{K}/{arm}_ver.jsonl"):
        j=json.loads(l); out[j['case_index']]={norm(r['y']):(r['y'],r['log_p_full']) for r in j['rows']}
    return out
cut=sys.argv[1] if len(sys.argv)>1 else 'vcut_zeroA'
A=rows('vcut_id'); B=rows(cut)
cases=sorted(i for i in A if i in B and len(A[i])==20 and len(B[i])==20)
classes=sorted(A[cases[0]].keys()); name={k:A[cases[0]][k][0] for k in classes}
ZA=np.array([[A[i][k][1] for k in classes] for i in cases]); ZB=np.array([[B[i][k][1] for k in classes] for i in cases])
D=ZA-ZB; n,C=D.shape
gold=[classes.index(norm(DS[i]['Answer'])) for i in cases]
print(f"### {cut}: D = {n}×{C}")
# SVD uncentered
U,s,Vt=np.linalg.svd(D,full_matrices=False); ev=s**2/np.sum(s**2)
print(f"SVD(uncentered) explained: PC1 {ev[0]:.2f}, PC2 {ev[1]:.2f}, PC3 {ev[2]:.2f}; PC1 direction top5: {[ (name[classes[j]][:8], round(float(Vt[0][j]),2)) for j in np.argsort(-np.abs(Vt[0]))[:5]]}")
mu=D.mean(0); Dc=D-mu
U2,s2,Vt2=np.linalg.svd(Dc,full_matrices=False); ev2=s2**2/np.sum(s2**2)
print(f"PCA(centered)   explained: PC1 {ev2[0]:.2f}, PC2 {ev2[1]:.2f}, PC3 {ev2[2]:.2f}")
print(f"||mu_V||={np.linalg.norm(mu):.2f} vs median ||eps_i||={np.median(np.linalg.norm(Dc,axis=1)):.2f}; share of total energy in mean: {np.sum(mu**2)*n/np.sum(D**2):.2f}")
cos=[float(D[i]@mu/(np.linalg.norm(D[i])*np.linalg.norm(mu)+1e-9)) for i in range(n)]
print(f"cos(Δz_i, mu_V): median {np.median(cos):+.2f}, q25 {np.percentile(cos,25):+.2f}, q75 {np.percentile(cos,75):+.2f}, >0.5: {sum(c>0.5 for c in cos)}/{n}, <0: {sum(c<0 for c in cos)}/{n}")
print("mu_V top/bottom classes:", [(name[classes[j]][:8],round(float(mu[j]),2)) for j in np.argsort(-mu)[:5]], '...', [(name[classes[j]][:8],round(float(mu[j]),2)) for j in np.argsort(-mu)[-3:]])
def stats(M,label):
    ranks=[]; dm=[]; dmc=[]
    for r,i in enumerate(cases):
        g=gold[r]; comp=max((j for j in range(C) if j!=g), key=lambda j: ZA[r][j])   # identity top non-gold
        others=np.delete(M[r],g); ranks.append(1+int(np.sum(others>M[r][g])))
        dm.append(M[r][g]-M[r][comp]); dmc.append(M[r][g]-np.max(others))
    ranks=np.array(ranks); dm=np.array(dm); dmc=np.array(dmc)
    print(f"{label}: gold rank median {np.median(ranks):.1f} (rank1 {np.sum(ranks==1)}, top3 {np.sum(ranks<=3)}, bottom half {np.sum(ranks>10)}) | gold−comp median {np.median(dm):+.2f} (>+0.5: {np.sum(dm>0.5)}, <−0.5: {np.sum(dm<-0.5)}) | gold−max(other) median {np.median(dmc):+.2f}, >0 in {np.sum(dmc>0)}/{n}")
stats(D,"raw Δz      ")
stats(Dc,"Δz⊥ = Δz−mu")
# remove PC1 (uncentered) instead of mean
P1=np.outer(D@Vt[0], Vt[0]); stats(D-P1,"Δz − PC1proj")
# class prior comparison
za=ZA.mean(0); zb=ZB.mean(0)
print("\n### class means: zeroA(text-ish prior) vs identity vs mu_V   [logP mean over cases]")
order=np.argsort(-zb)
print(f"{'class':14s} {'zeroA':>7s} {'identity':>8s} {'mu_V':>6s} {'rank_zeroA':>10s} {'rank_id':>7s}")
rank_b=np.argsort(np.argsort(-zb)); rank_a=np.argsort(np.argsort(-za))
for j in np.argsort(-mu):
    print(f"{name[classes[j]][:14]:14s} {zb[j]:7.2f} {za[j]:8.2f} {mu[j]:+6.2f} {rank_b[j]+1:10d} {rank_a[j]+1:7d}")
gc=np.bincount(gold,minlength=C)
print("gold class counts in gtex40:", [(name[classes[j]][:8],int(gc[j])) for j in np.argsort(-gc) if gc[j]>0])
print("corr(mu_V, -zeroA mean) =", round(float(np.corrcoef(mu,-zb)[0,1]),2), " corr(mu_V, gold freq) =", round(float(np.corrcoef(mu,gc)[0,1]),2))
