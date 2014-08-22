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


Dy2 = sp.spdiags([e, -2*e, e]/hy**2, [-1, 0, 1], Ny+1, Ny+1).todense()
Dy2[0, 0:3] = [1, -2, 1]/hy**2
Dy2[Ny, Ny-2:Ny+1] = [1,-2,1]/hy**2

Eta = -0.1*np.tanh((y-Ly/2)/Lj)
H = Eta + Hm
U = -g0/f0*np.dot(Dy,Eta)

etaB = 0*y

kk = np.arange(dkk,2+dkk,dkk)/Lj
#kk = kk[42]
nk = len(kk)

Z = np.zeros([Ny+1,Ny+1])
I = np.eye(Ny+1)
FI = f0*I

U1 = sp.spdiags(U, 0, Ny+1,Ny+1).todense()
H1 = sp.spdiags(H, 0, Ny+1,Ny+1).todense()
dU1 = sp.spdiags(np.transpose(np.dot(Dy,np.transpose(U))), 0, Ny+1,Ny+1).todense()
dH1 = sp.spdiags(np.dot(Dy,H), 0, Ny+1,Ny+1).todense()

grow = np.zeros([nEV,nk])
freq = np.zeros([nEV,nk])
mode = np.zeros([3*Ny+1,nEV,nk], dtype=complex)
# grOut = open('grow.dat', 'wb+')
# frOut = open('freq.dat', 'wb+')
# mdOut = open('mode.dat', 'wb+')
cnt = 0
guess=0.21+0.09*1j


A1 = np.concatenate((U1, dU1-FI, g0*I), axis=1)
A3 = np.concatenate((H1, dH1+H1*Dy,  U1), axis=1)

for kx in kk[0:91]: #0:nk

	#kx = kk[cnt]
	k2 = kx**2

	An = np.zeros([3*Ny+3, 3*Ny+3]) ##create numpy matrix, extra deleted later

	A2 = np.concatenate((-FI/k2, U1,-g0/k2*Dy), axis=1)
	An = np.concatenate((A1,A2,A3),axis=0)
	An = sc.delete(An, Ny+1, 0); An = sc.delete(An, 2*Ny, 0) #delete columns (middle is size Ny+1 x Ny-1 now)
	An = sc.delete(An, Ny+1, 1); An = sc.delete(An, 2*Ny, 1) #delete rows (middle is size Ny-1 x Ny-1 now)
	
	# # Using eig
	# eigVals, eigVecs = spalg.eig(An)
	# ind = (-np.imag(eigVals)).argsort() #get indices in descending order
	# eigVecs = eigVecs[:,ind]
	# eigVals = kx*eigVals[ind]
	# #print eigVals[:5]/kx

	# Using eigs
	evals_all, evecs_all = eigs(An,nEV,ncv=10,which='LI',sigma=guess,maxiter=5000)
	#print kx*evals_all[:]

	if len(evals_all)>5: evals=5
	else: evals = len(evals_all)
	for i in range(evals):
		grow[i,cnt] = np.imag(evals_all[i])*kx
		freq[i,cnt] = np.real(evals_all[i])*kx
		plt.plot(np.arange(0,nk),grow[i,:]*3600*24, 'o')
	cnt = cnt+1
	print cnt

ky = np.pi/Ly
plt.ylabel('1/day')
plt.xlabel('k/dk')
plt.title('Growth Rate: 1-Layer SW')
plt.savefig('Grow1L_SW.eps', format='eps', dpi=1000)
