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

## Creates A using numpy/scipy then moves rows to petsc4py matrix
## Could use eig, eigs, or slepc

rank = PETSc.COMM_WORLD.Get_rank()
Print = PETSc.Sys.Print
opts = PETSc.Options()
nEV = opts.getInt('nev', 5)
np.set_printoptions(threshold=np.nan)

Ly = 350e03
Lj = 20e03
Hm = 500
Ny = opts.getInt('Ny', 4)

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
nk = 1

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
grOut = open('grow.dat', 'wb+')
frOut = open('freq.dat', 'wb+')
mdOut = open('mode.dat', 'wb+')
cnt = 0
guess=1
newGuess=0

A1 = np.concatenate((U1, dU1-FI, g0*I), axis=1)
A3 = np.concatenate((H1, dH1+H1*Dy,  U1), axis=1)

for kx in kk[0:nk]:

	k2 = kx**2

	An = np.zeros([3*Ny+3, 3*Ny+3]) ##create numpy matrix, extra deleted later
	A2 = np.concatenate((-FI/k2, U1,-g0/k2*Dy), axis=1)
	An = np.concatenate((A1,A2,A3),axis=0)
	An = sc.delete(An, Ny+1, 0); An = sc.delete(An, 2*Ny, 0) #delete columns (middle is size Ny+1 x Ny-1 now)
	An = sc.delete(An, Ny+1, 1); An = sc.delete(An, 2*Ny, 1) #delete rows (middle is size Ny-1 x Ny-1 now)

	# print "\nNy=", Ny
	# print "\n" +"Eigenvalues with eig"
	# eigVals, eigVecs = spalg.eig(An)
	# ind = (-np.imag(eigVals)).argsort() #get indices in descending order
	# eigVecs = eigVecs[:,ind]
	# eigVals = kx*eigVals[ind]
	# print eigVals[:5]/kx

	# print "\n" +"Eigenvalues with eigs"
	# evals_all, evecs_all = eigs(An,which='LI',sigma=1,maxiter=1000)
	# print kx*evals_all[:]

	Ap = PETSc.Mat().create()
	Ap.setSizes([3*Ny+1, 3*Ny+1]); Ap.setFromOptions(); Ap.setUp()
	start,end = Ap.getOwnershipRange()

	for i in range(start,end): #change numpy matrix to petsc
		Ap[i,:] = An[i,:]
	Ap.assemble()

	#if cnt > 0: guess=newGuess
	E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)
	#E.setType(SLEPc.EPS.Type.LAPACK)
	#E.setBalance(2)
	sinv = E.getST()
	E.setOperators(Ap)
	E.setDimensions(nEV, PETSc.DECIDE)
	E.setProblemType(SLEPc.EPS.ProblemType.NHEP);E.setFromOptions()
	E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
	E.setTolerances(max_it=10000)

	sinv.setType('sinvert')
	sinv.setShift(guess)

	E.solve()

	nconv = E.getConverged()
	vr, wr = Ap.getVecs()
	vi, wi = Ap.getVecs()

	if nconv <= nEV: evals = nconv
	else: evals = nEV
	for i in range(evals):
		eigVal = E.getEigenvalue(i)
		grow[i,cnt] = eigVal.imag*kx
		freq[i,cnt] = eigVal.real*kx

		if rank == 0:
			grow[i,cnt].tofile(grOut)
			freq[i,cnt].tofile(frOut)
			plt.plot(np.arange(0,nk), grow[i]*3600*24, 'o')

		eigVec=E.getEigenvector(i,vr,vi)

		start,end = vi.getOwnershipRange()
		if start == 0: mode[0,i,cnt] = 0; start+=1
		if end == Ny: mode[Ny,i,cnt] = 0; end -=1

		for j in range(start,end):
			mode[j,i,cnt] = 1j*vi[j]; mode[j,i,cnt] += vr[j]
			if rank == 0:
				mode[j,i,cnt].tofile(mdOut)

	#print "\n" + "Eigenvalues with slepc"
	#print (freq[:,cnt]+grow[:,cnt]*1j)/kx
	#newGuess = (freq[0,cnt] + grow[0,cnt]*1j)/kx
	cnt = cnt+1

grOut.close(); frOut.close(); mdOut.close()
#Atest.close()
if rank == 0:
	ky = np.pi/Ly
	plt.ylabel('1/day')
	plt.xlabel('k')
	plt.title('Growth Rate: 1-Layer QG')
	plt.savefig('Grow1L_SW.eps', format='eps', dpi=1000)
	#plt.show()
