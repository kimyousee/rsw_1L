import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import scipy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.linalg as spalg
from scipy.sparse.linalg import eigs

## Code for using nump/scipy only (with eig or eigs)

opts = PETSc.Options()
nEV = opts.getInt('nev', 5)
np.set_printoptions(threshold=np.nan)

Ly = 350e03
Lj = 20e03
Hm = 500
Ny = opts.getInt('Ny', 400)

f0 = 1.e-4
bet = 0
g0 = 9.81
rho = 1024
dkk = 2e-2

y = np.linspace(0,Ly, Ny+1)
hy = y[1] - y[0]
e = np.ones(Ny+1)

Dy = sp.spdiags([-1*e, 0*e, e]/(2*hy), [-1, 0, 1], Ny+1,Ny+1).todense()
Dy[0, 0:2] = [-1, 1]/hy
Dy[Ny, Ny-1:Ny+1] = [-1, 1]/hy
sp.dia_matrix(Dy)

Dy2 = sp.spdiags([e, -2*e, e]/hy**2, [-1, 0, 1], Ny+1, Ny+1).todense()
Dy2[0, 0:3] = [1, -2, 1]/hy**2
Dy2[Ny, Ny-2:Ny+1] = [1,-2,1]/hy**2
sp.dia_matrix(Dy2)

Eta = -0.1*np.tanh((y-Ly/2)/Lj)
H = Eta + Hm
U = -g0/f0*np.dot(Dy,Eta)
U = np.transpose(U)
dU = np.dot(Dy,U)
dH = np.dot(Dy,H)
H1 = sp.spdiags(H, 0, Ny+1,Ny+1)
HDy = H1*Dy

kk = np.arange(dkk,2+dkk,dkk)/Lj
nk = len(kk)

grow = np.zeros([nEV,nk])
freq = np.zeros([nEV,nk])
mode = np.zeros([3*Ny+1,nEV,nk], dtype=complex)
# grOut = open('grow.dat', 'wb+')
# frOut = open('freq.dat', 'wb+')
# mdOut = open('mode.dat', 'wb+')

A1 = np.zeros([3*Ny+1,3*Ny+1])

A1[0,0]   = U[0]	## Block A00 corners
A1[Ny,Ny] = U[Ny]	##

A1[0, 2*Ny] = g0 	## Block A02 corners
A1[Ny,3*Ny] = g0 

A1[2*Ny,0] = H[0]	## Block A20 corners
A1[3*Ny,Ny] = H[Ny]

A1[2*Ny,2*Ny] = U[0]## Block A22 corners
A1[3*Ny,3*Ny] = U[Ny]

# Block A21 top 2 and bottom 2 corners
A1[2*Ny,   Ny+1] = 			HDy[0,1]
A1[2*Ny+1, Ny+1:Ny+3] = 	[dH[0,1],0] + HDy[1,1:3]
A1[3*Ny-1, 2*Ny-2:2*Ny] = 	[0,dH[0,Ny-1]] + HDy[Ny-1,Ny-2:Ny]
A1[3*Ny,   2*Ny-1] = 		HDy[Ny,Ny-1]

for ii in xrange(1,Ny):
	A1[ii,ii]      = U[ii]
	A1[ii,Ny+ii]   = dU[ii] - f0
	A1[ii,2*Ny+ii] = g0

	A1[ii+Ny,ii+Ny] = U[ii]

	A1[ii+2*Ny,ii]  = H[ii]
	A1[ii+2*Ny,2*Ny+ii] = U[ii]

	if ii == 1 or ii == (Ny-1):
		continue
	A1[ii+2*Ny, Ny+ii-1:Ny+ii+2] = [0,dH[0,ii],0] + HDy[ii,ii-1:ii+2]
sp.dia_matrix(A1)

A2 = np.zeros([3*Ny+1, 3*Ny+1])
for ii in xrange(1,Ny):
	A2[Ny+ii,ii] = -f0
	A2[Ny+ii,2*Ny+ii-1:2*Ny+ii+2] = -g0*Dy[ii,ii-1:ii+2]
sp.dia_matrix(A2)

evecOut = open('eigVec.bin', 'wb+')
evalOut = open('eigVal.bin', 'wb+')

cnt = 0
guess=0.21+0.09*1j

for kx in kk[0:nk]: #0:91... might not converge near the end

	k2 = kx**2

	A = A1+A2/k2
	sp.dia_matrix(A)

	# # Using eig
	# eigVals, eigVecs = spalg.eig(A)
	# ind = (-np.imag(eigVals)).argsort() #get indices in descending order
	# eigVecs = eigVecs[:,ind]
	# eigVals = kx*eigVals[ind]
	# #print eigVals[:5]/kx

	# Using eigs
	evals_all, evecs_all = eigs(A,nEV,ncv=10,which='LI',sigma=guess,maxiter=5000)
	#print kx*evals_all[:]
	evals_all.tofile(evalOut)
	evecs_all.tofile(evecOut)
	print evecs_all.shape,evals_all.shape
	print evals_all[:]
	if len(evals_all)>nEV: evals=nEV
	else: evals = len(evals_all)
	for i in xrange(evals):
		grow[i,cnt] = np.imag(evals_all[i])*kx
		freq[i,cnt] = np.real(evals_all[i])*kx
		plt.plot(np.arange(0,nk),grow[i,:]*3600*24, 'o')
	cnt = cnt+1


ky = np.pi/Ly
plt.ylabel('1/day')
plt.xlabel('k/dk')
plt.title('Growth Rate: 1-Layer SW2')
plt.savefig('Grow1L_SW.eps', format='eps', dpi=1000)
