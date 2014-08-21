import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import scipy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os

## Code for using petsc4py and slepc4py. This code uses a fixed guess
## -eps_target 0.23+0.074i or 0.23,0.074
OutpDir = "storage"
if not os.path.exists(OutpDir):
    os.mkdir(OutpDir)
data = open('storage/InputData','wb')
kkFile = open('storage/kk','wb')
evalsFile = open('storage/evals','wb')

rank = PETSc.COMM_WORLD.Get_rank()
Print = PETSc.Sys.Print
opts = PETSc.Options()
nEV = opts.getInt('nev', 2)

Ly = 350e03
Lj = 20e03
Hm = 500
Ny = opts.getInt('Ny', 300)

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
mode = np.zeros([3*Ny+1,nEV,nk], dtype=complex)
grOut = open('storage/grow', 'wb+')
frOut = open('storage/freq', 'wb+')
mdOut = open('storage/mode', 'wb+')

Ap = PETSc.Mat().createAIJ([3*Ny+1, 3*Ny+1],comm=PETSc.COMM_WORLD)
Ap.setFromOptions(); Ap.setUp()
start,end = Ap.getOwnershipRange()

# Set up all parts of A except Blocks A10 and A12 (set up the ones that don't use kx)
for i in range(start,end):
    if (0 <= i <= Ny): #first row; Ny+1
        Ap[i,i] = U[i]
        Ap[i, 2*Ny+i] = g0 #Block A02
        if (1 <= i <= Ny-1): # top row and bottom of block is all 0 for A01
            Ap[i, Ny+i] = dU[i] - f0 #Block A01

    if Ny <= i <= 2*Ny: #2nd row; Ny-1
        Ap[i,i] = U[i-Ny]

    if 2*Ny <= i <= 3*Ny: #3rd row
        cols1 = i-2*Ny  #1 - Ny
        cols2 = i-Ny+1  #Ny+1 - 2*Ny
        Ap[i,i] = U[cols1]
        Ap[i,cols1] = H[cols1] #Block A20
        #Block A21
        if (cols1 == 0): #assign only 1 value (top corner of block)
            Ap[i,cols2] = HDy[0,1]
        elif (cols1 == Ny): #bottom corner
            Ap[i,cols2-2] = HDy[Ny, Ny-1]
        elif (cols1 == 1): #2 elements top
            Ap[i,[cols2-1,cols2]] = [dH[1], HDy[1,2]]
        elif (cols1 == Ny-1): #2 elements bottom
            Ap[i,[cols2-2,cols2-1]] = [0,dH[Ny-1]] + HDy[Ny-1,Ny-2:Ny].todense()
        elif (2 <= cols1 <= Ny-2): #3 elements
            Ap[i,cols2-2:cols2+1] = [HDy[cols1, cols1-1], dH[cols1], HDy[cols1, cols1+1]]
Ap.assemble()

evalsArr = np.zeros(nk)
cnt = 0
# guess=freq[0,49]/kx + grow[0,49]*1j/kx
guess = 0.21+0.09*1j

for kx in kk[0:nk]: #0:nk
    k2 = kx**2

    A = PETSc.Mat().createAIJ([3*Ny+1,3*Ny+1])
    A.setFromOptions(); A.setUp()
    A = Ap.copy()
    start,end = A.getOwnershipRange()

    for i in xrange(start,end):
        if Ny+1 <= i < 2*Ny:
            cols1 = i-Ny # goes from 1-Ny
            cols3 = i+Ny-1 #goes from Ny+1 :
            A[i,cols1] = -f0/k2 # Block A10
            A[i,cols3:cols3+3] = (-g0/k2)*Dy[cols1,cols1-1:cols1+2].todense() # Block A12
    A.assemble()

    #if cnt > 0: guess=newGuess

    E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)
    sinv = E.getST()
    E.setOperators(A); E.setDimensions(nEV, SLEPc.DECIDE)
    #E.setType(SLEPc.EPS.Type.LAPACK)
    # E.setBalance(2) #SLEPc.EPS.Balance.ONESIDE
    E.setProblemType(SLEPc.EPS.ProblemType.NHEP);E.setFromOptions()
    E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
    E.setTolerances(1e-8,max_it=200)
    
    sinv.setType('sinvert')
    sinv.setShift(guess)

    E.solve()

    nconv = E.getConverged()
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    if nconv <= nEV: evals = nconv
    else: evals = nEV
    evalsArr[cnt] = evals

    for i in xrange(evals):
        eigVal = E.getEigenvalue(i)
        grow[i,cnt] = eigVal.imag*kx
        freq[i,cnt] = eigVal.real*kx

        eigVec=E.getEigenvector(i,vr,vi)

        start,end = vi.getOwnershipRange()
        if start == 0: mode[0,i,cnt] = 0; start+=1
        if end == Ny: mode[Ny,i,cnt] = 0; end -=1

        for j in xrange(start,end):
            mode[j,i,cnt] = 1j*vr[j].imag + vr[j].imag
    Print(cnt,kx,(freq[:,cnt]+grow[:,cnt]*1j)/kx)
    cnt = cnt+1

grow.tofile(grOut)
freq.tofile(frOut)
mode.tofile(mdOut)
dataArr.tofile(data)
kk.tofile(kkFile)
evalsArr.tofile(evalsFile)
data.close(); grOut.close();frOut.close();mdOut.close();kkFile.close()
evalsFile.close()

