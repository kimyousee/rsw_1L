import numpy as np
import scipy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.linalg as spalg
from scipy.sparse.linalg import eigs
import os

## Code for using numpy/scipy only (with eig or eigs)

OutpDir = "storage"
if not os.path.exists(OutpDir):
    os.mkdir(OutpDir)
data = open('storage/InputData','wb')
kkFile = open('storage/kk','wb')
evalsFile = open('storage/evals','wb')

nEV = 5

Ly = 350e03
Lj = 20e03
Hm = 500
Ny = 100

f0 = 1.e-4
bet = 0
g0 = 9.81
rho = 1024
dkk = 2e-2
dataArr = np.array([Ly,Lj,Hm,Ny,f0,bet,g0,rho,dkk,nEV])

y = np.linspace(0,Ly, Ny+1)
hy = y[1] - y[0]
e = np.ones(Ny+1)

Dy = sp.spdiags([-1*e, 0*e, e]/(2*hy), [-1, 0, 1], Ny+1,Ny+1)
Dy = sp.csr_matrix(Dy)
# Dy = sp.lil_matrix(Dy)
Dy[0, 0:2] = [-1, 1]/hy
Dy[Ny, Ny-1:Ny+1] = [-1, 1]/hy

Dy2 = sp.spdiags([e, -2*e, e]/hy**2, [-1, 0, 1], Ny+1, Ny+1)
# Dy2 = sp.lil_matrix(Dy2)
Dy2 = sp.csr_matrix(Dy2)
Dy2[0, 0:3] = [1, -2, 1]/hy**2
Dy2[Ny, Ny-2:Ny+1] = [1,-2,1]/hy**2

Eta = -0.1*np.tanh((y-Ly/2)/Lj)
H = Eta + Hm
U = -g0/f0*Dy*Eta
U = np.transpose(U)

dU = Dy*U
dH = Dy*H
H1 = sp.spdiags(H, 0, Ny+1,Ny+1)
H1 = sp.csr_matrix(H1)
HDy = H1*Dy

kk = np.arange(dkk,2+dkk,dkk)/Lj
nk = len(kk)

grow = np.zeros([nEV,nk])
freq = np.zeros([nEV,nk])
mode = np.zeros([3*Ny+1,nEV,nk], dtype=np.complex128)
grOut = open('storage/grow', 'wb+')
frOut = open('storage/freq', 'wb+')
mdOut = open('storage/mode', 'wb+')


# A1 = np.zeros([3*Ny+1,3*Ny+1])
A1 = sp.csr_matrix((3*Ny+1,3*Ny+1))

A1[0,0]   = U[0]    ## Block A00 corners
A1[Ny,Ny] = U[Ny]   ##

A1[0, 2*Ny] = g0    ## Block A02 corners
A1[Ny,3*Ny] = g0 

A1[2*Ny,0] = H[0]   ## Block A20 corners
A1[3*Ny,Ny] = H[Ny]

A1[2*Ny,2*Ny] = U[0]## Block A22 corners
A1[3*Ny,3*Ny] = U[Ny]

# Block A21 top 2 and bottom 2 corners
A1[2*Ny,   Ny+1] =          HDy[0,1]
A1[2*Ny+1, Ny+1:Ny+3] =     [dH[1],0] + HDy[1,1:3].todense()
A1[3*Ny-1, 2*Ny-2:2*Ny] =   [0,dH[Ny-1]] + HDy[Ny-1,Ny-2:Ny].todense()
A1[3*Ny,   2*Ny-1] =        HDy[Ny,Ny-1]

for ii in range(1,Ny):
    A1[ii,ii]      = U[ii]
    A1[ii,Ny+ii]   = dU[ii] - f0
    A1[ii,2*Ny+ii] = g0

    A1[ii+Ny,ii+Ny] = U[ii]

    A1[ii+2*Ny,ii]  = H[ii]
    A1[ii+2*Ny,2*Ny+ii] = U[ii]

    if ii == 1 or ii == (Ny-1):
        continue
    A1[ii+2*Ny, Ny+ii-1:Ny+ii+2] = [0,dH[ii],0] + HDy[ii,ii-1:ii+2].todense()


# A2 = np.zeros([3*Ny+1, 3*Ny+1])
A2 = sp.csr_matrix((3*Ny+1, 3*Ny+1))
for ii in range(1,Ny):
    A2[Ny+ii,ii] = -f0
    A2[Ny+ii,2*Ny+ii-1:2*Ny+ii+2] = -g0*Dy[ii,ii-1:ii+2]

evalsArr = np.zeros(nk)
cnt = 0
guess=0.21+0.09*1j

for kx in kk[0:nk]: #0:nk

    k2 = kx**2

    A = A1+A2/k2
    sp.dia_matrix(A)

    # Using eig
    eigVals, eigVecs = spalg.eig(A.todense())
    ind = (-np.imag(eigVals)).argsort() #get indices in descending order
    eigVecs = eigVecs[:,ind]
    eigVals = kx*eigVals[ind]

    # #print eigVals[:5]/kx

    # Using eigs
    # evals_all, evecs_all = eigs(A,nEV,ncv=10,which='LI',sigma=guess,maxiter=5000)

    if len(eigVals)>nEV: evals=nEV
    else: evals = len(eigVals)
    evalsArr[cnt] = evals

    for i in range(evals):
        grow[i,cnt] = np.imag(eigVals[i])*kx
        freq[i,cnt] = np.real(eigVals[i])*kx
        for j in range(eigVals.shape[0]):
            mode[j,i,cnt] = eigVecs[j,i]
    cnt = cnt+1

grow.tofile(grOut)
freq.tofile(frOut)
mode.tofile(mdOut)
dataArr.tofile(data)
kk.tofile(kkFile)
evalsArr.tofile(evalsFile)
data.close(); grOut.close();frOut.close();mdOut.close();kkFile.close()
evalsFile.close()

