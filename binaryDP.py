## Reproduce results from the paper:
# Freirich, Dror, Nir Weinberger, and Ron Meir. "The Distortion-Perception Tradeoff in Finite Channels with Arbitrary Distortion Measures." NeurIPS 2023 workshop: Information-Theoretic Principles in Cognitive Systems. 2023.‏

import numpy as np
import cvxpy as cp
g_cvx_eps = 1e-11
pnorm = 1

from tqdm import tqdm,  trange

import matplotlib
from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

## simulation parameters
Y = 6
X = 2
yus0 = Y - 2
dP = 5750

g_seed = 1337
np.random.seed(   g_seed)

def D_hamming(X):
    return np.float64(1. - np.eye(X))

# random distortion measure
D = np.random.randn(X,X)
D -= np.min(D)
D /= np.max(D)

# TV perception index
H = D_hamming(X)

np.random.uniform(0,100,1+3+2*Y)

# ramdomize channel
Pxgy = np.ones(shape=(X,Y),dtype = np.float64)
for r in range(1,yus0+1):
    Pxgy[:,-r] = Pxgy[:,-r]*0 + np.random.uniform(0,1,(X,))*1

Pxgy /= np.sum(Pxgy, axis = 0, keepdims = True)

# source and output distributions
Py  = np.float64( np.random.uniform(0,1,(Y,1)) )
Py /= np.sum(Py)
Px  = Pxgy@Py
Pxandy = Pxgy*Py.T
Pxy = (Pxandy / np.tile( Px, Y))

Pxy /= np.sum(Pxy, axis = 1, keepdims = True)

Py   = Pxy.T@Px
rho  = D.T@(Pxy*Px)
rho_ = rho/Py.T

## analytical solution (a-la Theorem 4.2)
fig, ax = plt.subplots(1)
Dstar = np.sum( np.min(rho, axis  =0) )

Us = []
for y in range(Y):
    u1 = rho_[0,y]-rho_[1,y]
    u2 = -u1/H[1,0]
    u1 = u1 if np.abs(u1)>1e-9 else 0.
    u2 = u2 if np.abs(u2)>1e-11 else 0.

    Us.append(u1/2)


Us   = np.array(Us)
Yord = np.argsort(Us)

Us   = Us[Yord]
Pxy  = Pxy[:,Yord]
Py   = Pxy.T@Px
rho  = D.T@(Pxy*Px)
rho_ = rho/Py.T

Py_  = np.sum([Py[y,0] if Us[y]<=0 else 0. for y in range(Y)])
Py_1 = np.sum([Py[y,0] if Us[y]<0 else 0. for y in range(Y)])

direct = 1 if (Px[0,0]-Py_) >= 0 else (-1 if (Px[0,0]-Py_1) <= 0 else 0)

p_ = Px[0,0]-Py_ if direct ==1 else -(Px[0,0]-Py_1)  #if px1geqpy_ else -(Px[0]-Py_)
p_ = np.max([p_,0])
d_ = Dstar

if not direct == 0:
    ui = np.where(direct*Us > 0)[0][0 if direct>0 else -1]


PDS=[]
while p_ > 1e-20:
    PDS.append([p_,d_])

    if direct >0:
        p = Px[0,0] - np.sum([Py[y,0] if Us[y]<=Us[ui] else 0. for y in range(Y)])

    if direct <0:
        p = np.sum([Py[y,0] if Us[y]<Us[ui] else 0. for y in range(Y)]) -  Px[0,0]

    p = np.max([p,0])
    d = d_ + 2*np.abs(Us[ui])*(p_-p)

    d_=d ; p_ = p
    ui += direct

# ax.scatter(p_,d_,s=15,c='k')
PDS.append([p_,d_])

## numerical solution
x1 = np.ones([1,X])
y1 = np.ones([1,Y])


DPs = []

Q  = cp.Variable((X,Y), symmetric=False)
PI = cp.Variable((X,X), symmetric=False)
P = np.linspace(0,1,num = int(dP) + 1)

constraints0 = []
constraints0 += [ x1@Q == y1 ]
constraints0 += [ x1@PI  == Px.T ]
constraints0 += [ PI@x1.T  == Q@Py ]

constraints0 += [ Q >= 0 ]
constraints0 += [ PI >= 0 ]

for Pi in tqdm(P):
    constraints = constraints0 + [ cp.trace(H.T@PI) <= Pi ]

    prob = cp.Problem(cp.Minimize(cp.trace(rho.T@Q)), constraints)
    prob.solve(solver=cp.SCS,verbose=False,warm_start=True, eps = g_cvx_eps, max_iters = 10000000)

    Qv = Q.value
    Pv = cp.norm(Px-Q@Py,1).value
    DP = np.trace(rho.T@Qv)

    DPs.append(DP)

P   = np.array(P)
DPs = np.array(DPs)

pind = np.where(P < 0.8)[0]
ax.plot(P[pind],DPs[pind],lw=4)

pnum = 0
ep = 1.0075
xt = [1.,0]
for p_,d_ in PDS:
    if pnum == 0:
        ax.annotate(r'$D^*$', (.5+.5*p_,d_*ep), fontsize=24)
        dstar = d_
        pstar = p_

    if pnum == 1:
        ax.annotate(r'slope $2u_1$', (p_*.35,d_*.5 + dstar*.5), fontsize=24)
        plt.annotate(r'', xytext=(p_, dstar), xy=(p_, d_) ,
horizontalalignment="center",
            arrowprops=dict(arrowstyle='fancy', color='orange', lw=.35))

        plt.annotate(r'', xy=(pstar, dstar), xytext=(p_, dstar) ,
horizontalalignment="center",
            arrowprops=dict(arrowstyle='-',linestyle=":", color='k', lw=2))

    else:
        plt.vlines(p_, dstar, d_, alpha=0.54,colors='k', ls = ':',zorder=1)

    ax.scatter(p_, d_, s=30, c='k', zorder=3)
    if p_ > 1e-11:
        ax.annotate(r'$P^*_' + str(pnum) + '$', (p_*ep,d_*ep), fontsize=24)
        pnum+=1
# for p_,d_ ,_,_ in qPDS:
#     ax.scatter(p_,d_,s=15,c='g',marker='d')

plt.hlines(dstar, 0.8, 1., ls='--', colors='b', lw = 4)
ax.fill_between(P[pind],y1 =DPs[pind] , y2=dstar,alpha=.35)
plt.text(x = pstar/28,y=d_/28+dstar,s='(unattainable region)', rotation='vertical')

plt.xlabel(r'$P$', fontsize=24)
plt.ylabel(r'$D(P)$', fontsize=18)

plt.yticks([])
plt.xticks(xt)

plt.title(direct)
plt.show()
