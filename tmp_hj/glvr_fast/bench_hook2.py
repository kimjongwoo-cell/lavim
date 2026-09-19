"""Variants of the decode-step hook math; all must agree numerically."""
import time, torch
torch.manual_seed(0)
dev="cuda"; H,Hkv,D,Hid,L,base,Nd,m=32,8,128,2560,5000,4600,3850,10; G=H//Hkv
qp=torch.nn.Linear(Hid,H*D,bias=False,device=dev,dtype=torch.bfloat16)
op=torch.nn.Linear(H*D,Hid,bias=False,device=dev,dtype=torch.bfloat16)
qn=torch.nn.RMSNorm(D,device=dev,dtype=torch.bfloat16)
K=torch.randn(Hkv,L,D,device=dev,dtype=torch.bfloat16); V=torch.randn(Hkv,L,D,device=dev,dtype=torch.bfloat16)
don=torch.randperm(base,device=dev)[:Nd].sort().values; lat=torch.arange(4000,4010,device=dev)
sel=torch.cat([don,lat])
Kp=K[:,:base].float(); Kn=K[:,base:].float(); Vd=V.index_select(1,don).float(); U=torch.randn(H,m,D,device=dev)
Kall=torch.cat([Kp,Kn],1).contiguous()
def rot(q,cos,sin):
    x1,x2=q[...,:D//2],q[...,D//2:]
    return q*cos+torch.cat([-x2,x1],-1)*sin
def mk(R):
    h=torch.randn(1,R,Hid,device=dev,dtype=torch.bfloat16); cos=torch.randn(1,1,R,D,device=dev,dtype=torch.bfloat16)
    return h,cos,cos
def base_v(h,cos,sin,R):
    q=rot(qn(qp(h).view(1,R,H,D)).transpose(1,2),cos,sin)
    qg=q[0].float().reshape(Hkv,G*R,D)
    s=torch.cat([qg@Kp.transpose(-2,-1), qg@Kn.transpose(-2,-1)],-1).view(H,R,L)*0.088
    A=torch.softmax(s,-1); Ad=A.index_select(-1,don); rho=Ad.sum(-1,keepdim=True)
    mB=torch.einsum("kgrn,knd->kgrd",Ad.view(Hkv,G,R,-1),Vd).reshape(H,R,D)
    al=A.index_select(-1,lat); g=al/al.sum(-1,keepdim=True); u=torch.einsum("hrm,hmd->hrd",g,U)
    return op((0.75*(rho*u-mB)).permute(1,0,2).reshape(1,R,-1).to(torch.bfloat16))
def opt_v(h,cos,sin,R):
    q=rot(qn(qp(h).view(1,R,H,D)).transpose(1,2),cos,sin)
    qg=q[0].float().reshape(Hkv,G*R,D)
    s=torch.bmm(qg,Kall.transpose(-2,-1)).view(H,R,L).mul_(0.088)
    A=torch.softmax(s,-1)
    As=A.index_select(-1,sel)                       # one gather for donor + latent
    Ad=As[...,:Nd]; al=As[...,Nd:]
    rho=Ad.sum(-1,keepdim=True)
    mB=torch.bmm(Ad.view(Hkv,G*R,Nd),Vd).view(H,R,D)
    g=al/al.sum(-1,keepdim=True); u=torch.bmm(g,U)
    return op((rho*u-mB).mul_(0.75).permute(1,0,2).reshape(1,R,-1).to(torch.bfloat16))
h,c,s_=mk(1)
a,b=base_v(h,c,s_,1),opt_v(h,c,s_,1)
print("max abs diff", float((a.float()-b.float()).abs().max()), "rel", float((a.float()-b.float()).norm()/a.float().norm()))
for name,fn in (("base",base_v),("opt",opt_v)):
    for R,n in ((1,300),(350,20)):
        h,c,s_=mk(R)
        for _ in range(5): fn(h,c,s_,R)
        torch.cuda.synchronize(); t=time.time()
        for _ in range(n): fn(h,c,s_,R)
        torch.cuda.synchronize(); print(f"{name} R={R}: {(time.time()-t)/n*1e3:.2f} ms")
