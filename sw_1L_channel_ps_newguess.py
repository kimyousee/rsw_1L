import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import scipy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

## Code for changing guess to be the previous one just computed.
## Starts at 84 then goes up to 200 (or 199)
## Then goes back to 83 and goes down to 0
## Note: code is really long because it is duplicated; will fix later.

rank = PETSc.COMM_WORLD.Get_rank()
Print = PETSc.Sys.Print
opts = PETSc.Options()
nEV = opts.getInt('nev', 5)

Ly = 350e03
Lj = 20e03
Hm = 500
Ny = opts.getInt('Ny', 300)

f0 = 1.e-4
bet = 0
g0 = 9.81
rho = 1024
dkk = 1e-2 #2e-2

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
grOut = open('grow.dat', 'wb+')
frOut = open('freq.dat', 'wb+')
mdOut = open('mode.dat', 'wb+')

Ap = PETSc.Mat().create()
Ap.setSizes([3*Ny+1, 3*Ny+1]); Ap.setFromOptions(); Ap.setUp()
start,end = Ap.getOwnershipRange()

for i in xrange(start,end):
	if (0 <= i <= Ny): #first row; Ny+1
		Ap[i,i] = U[i]
		Ap[i, 2*Ny+i] = g0 #Block A02
		if (1 <= i <= Ny-1): # top row and bottom of block is all 0 for A01
			Ap[i, Ny+i] = dU[i] - f0 #Block A01

	if Ny <= i <= 2*Ny: #2nd row; Ny-1
		Ap[i,i] = U[i-Ny]

	if 2*Ny <= i <= 3*Ny: #3rd row
		cols1 = i-2*Ny 	#1 - Ny
		cols2 = i-Ny+1	#Ny+1 - 2*Ny
		Ap[i,i] = U[cols1]
		Ap[i,cols1] = H[cols1] #Block A20
		#Block A21
		if (cols1 == 0): #assign only 1 value (top corner of block)
			Ap[i,cols2] = HDy[0,1]
		elif (cols1 == Ny): #bottom corner
			Ap[i,cols2-2] = HDy[Ny, Ny-1]
		elif (cols1 == 1): #2 elements top
			Ap[i,[cols2-1,cols2]] = [dH[0,1], HDy[1,2]]
		elif (cols1 == Ny-1): #2 elements bottom
			Ap[i,[cols2-2,cols2-1]] = [0,dH[0,Ny-1]] + HDy[Ny-1,Ny-2:Ny]
		elif (2 <= cols1 <= Ny-2): #3 elements
			Ap[i,cols2-2:cols2+1] = [HDy[cols1, cols1-1], dH[0,cols1], HDy[cols1, cols1+1]]
Ap.assemble()


cnt = 84 #42#50
guess1 = 0.22+0.091*1j # Guess at 42
newGuess = guess1

for kx in kk[84:nk]: # Start at 84 and go right
	kx = kk[cnt]
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
			A[i,cols3:cols3+3] = (-g0/k2)*Dy[cols1,cols1-1:cols1+2] # Block A12
	A.assemble()

	if cnt > 84: guess=newGuess
	else: guess = guess1

	E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)
	sinv = E.getST()
	E.setOperators(A); E.setDimensions(nEV, SLEPc.DECIDE)
	#E.setTarget(0.55203)
	#E.setType(SLEPc.EPS.Type.LAPACK)
	E.setBalance(2) #SLEPc.EPS.Balance.ONESIDE
	E.setProblemType(SLEPc.EPS.ProblemType.NHEP);E.setFromOptions()
	E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
	E.setTolerances(max_it=500)
	
	sinv.setType('sinvert')
	sinv.setShift(guess)

	E.solve()

	nconv = E.getConverged()
	vr, wr = A.getVecs()
	vi, wi = A.getVecs()

	if nconv <= nEV: evals = nconv
	else: evals = nEV

	for i in xrange(evals):
		eigVal = E.getEigenvalue(i)
		grow[i,cnt] = eigVal.imag*kx
		freq[i,cnt] = eigVal.real*kx
		
		if rank == 0:
			grow[i,cnt].tofile(grOut)
			freq[i,cnt].tofile(frOut)
			plt.plot(np.arange(0,nk), grow[i,:]*3600*24, 'o')

		eigVec=E.getEigenvector(i,vr,vi)

		start,end = vi.getOwnershipRange()
		if start == 0: mode[0,i,cnt] = 0; start+=1
		if end == Ny: mode[Ny,i,cnt] = 0; end -=1

		for j in xrange(start,end):
			mode[j,i,cnt] = 1j*vi[j]; mode[j,i,cnt] = vr[j]
			if rank == 0:
				mode[j,i,cnt].tofile(mdOut)

	Print(cnt, grow[:,cnt])
	#print freq[:,cnt]+grow[:,cnt]*1j
	if (freq[0,cnt]+grow[0,cnt]*1j)/kx!=0: newGuess = (freq[0,cnt] + grow[0,cnt]*1j)/kx
	else: newGuess = guess
	cnt = cnt+1

cnt=83
newGuess=guess1

for kx in kk[83::-1]: # Start at 83 and go left
	kx = kk[cnt]
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
			A[i,cols3:cols3+3] = (-g0/k2)*Dy[cols1,cols1-1:cols1+2] # Block A12
	A.assemble()

	if cnt < 83: guess=newGuess
	else: guess = guess1

	E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)
	sinv = E.getST()
	E.setOperators(A); E.setDimensions(nEV, SLEPc.DECIDE)
	#E.setTarget(0.55203)
	#E.setType(SLEPc.EPS.Type.LAPACK)
	E.setBalance(2) #SLEPc.EPS.Balance.ONESIDE
	E.setProblemType(SLEPc.EPS.ProblemType.NHEP);E.setFromOptions()
	E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
	E.setTolerances(max_it=500)
	
	sinv.setType('sinvert')
	sinv.setShift(guess)

	E.solve()

	nconv = E.getConverged()
	vr, wr = A.getVecs()
	vi, wi = A.getVecs()

	if nconv <= nEV: evals = nconv
	else: evals = nEV

	for i in xrange(evals):
		eigVal = E.getEigenvalue(i)
		grow[i,cnt] = eigVal.imag*kx
		freq[i,cnt] = eigVal.real*kx
		
		if rank == 0:
			grow[i,cnt].tofile(grOut)# print >>grOut, grow[i,cnt]
			freq[i,cnt].tofile(frOut)# print >>frOut, freq[i,cnt]
			plt.plot(np.arange(0,nk), grow[i,:]*3600*24, 'o')

		eigVec=E.getEigenvector(i,vr,vi)

		start,end = vi.getOwnershipRange()
		if start == 0: mode[0,i,cnt] = 0; start+=1
		if end == Ny: mode[Ny,i,cnt] = 0; end -=1

		for j in xrange(start,end):
			mode[j,i,cnt] = 1j*vi[j]; mode[j,i,cnt] = vr[j]
			if rank == 0:
				mode[j,i,cnt].tofile(mdOut)#print >>mdOut, mode[j, i, cnt]

	#print freq[:,cnt]+grow[:,cnt]*1j
	eigVal1 = (freq[0,cnt]+grow[0,cnt]*1j)/kx
	if eigVal1 > 1e-10: # if eigVal is non-zero
		newGuess = eigVal1
	else: newGuess = guess #previous non-zero guess
	Print(cnt, grow[:,cnt])
	cnt = cnt-1
	

grOut.close(); frOut.close(); mdOut.close()
if rank == 0:
	ky = np.pi/Ly
	plt.ylabel('1/day')
	plt.xlabel('k/dk')
	plt.title('Growth Rate: 1-Layer SW')
	plt.savefig('Grow1L_SW.eps', format='eps', dpi=1000)
	#plt.show()
