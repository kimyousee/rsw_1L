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

## Testing to use matrix-free/shell matrices for petsc4py

def rsw1L(M,x,f,Ny,k2,U,dU,H,dH,hy,f0=1.e-4,g0=9.81):

    M[:] = x

    dU = np.ravel(dU)

    #Break up each part of x (or M)
    u = np.ravel(M[0:Ny+1])         #size Ny+1
    v = np.ravel(M[Ny+1:2*Ny])      #size Ny-1
    h = np.ravel(M[2*Ny:3*Ny+1])    #size Ny+1
    w = np.hstack([0,v,0]) #add 0s at endpoints

    Dyh = np.hstack([(h[1]-h[0])/hy, (h[2:Ny+1]-h[0:Ny-1])/(2*hy), (h[Ny]-h[Ny-1])/hy])
    Dyv = np.hstack([(w[1]-w[0])/hy, (w[2:Ny+1]-w[0:Ny-1])/(2*hy), (w[Ny]-w[Ny-1])/hy])

    f[0:Ny+1]       = U*u + (dU-f0)*w + g0*h
    f[Ny+1:2*Ny]    = -f0/k2*u[1:Ny] + U[1:Ny]*v - g0/k2*Dyh[1:Ny]
    f[2*Ny:3*Ny+1]  = H*u + dH*w + H*Dyv + U*h

class RSW1L(object):
	def __init__(self,Ny,k2,U,dU,H,dH,hy,f0,g0):
		self.size = 3*Ny+1
		self.M  = np.zeros([self.size],dtype=np.complex64)
		self.Ny	= Ny
		self.U,self.dU = U,dU
		self.H,self.dH = H,dH
		self.k2 = k2
		self.hy = hy
		self.f0 = f0
		self.g0 = g0

	def mult (self,A,x,y):
		self.xx = x[...].reshape(self.size) #[ui,vi,hi]
		self.yy = y[...].reshape(self.size) #result vector
		rsw1L(self.M, self.xx, self.yy, self.Ny, self.k2,
			self.U, self.dU, self.H, self.dH, self.hy)
	def getDiagonal(self,A,diag):
		U = self.U
		diag = np.hstack([U,U[1:Ny],U])

def construct_operator(Ny,k2,U,dU,H,dH,hy,f0,g0):
	context = RSW1L(Ny,k2,U,dU,H,dH,hy,f0,g0)
	A = PETSc.Mat().createPython([3*Ny+1,3*Ny+1],context)
	A.setUp()
	return A

def solve_eigensystem(Ap,guess,nEV,cnt,freq,grow,mode,problem_type=SLEPc.EPS.ProblemType.NHEP):
	E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)
	sinv = E.getST()

	E.setOperators(Ap)
	E.setDimensions(nEV, PETSc.DECIDE)
	E.setProblemType(problem_type)
	E.setTarget(guess)
	#E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
	E.setTolerances(1e-5, max_it=100)

	# sinv.setMatMode(2)			#st_matmode shell
	# sinv.setType('sinvert') 	#st_type sinvert
	# sinv.setShift(guess)		#st_shift ___
	
	E.setFromOptions()
	E.view()
	
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
			# grow[i,cnt].tofile(grOut)
			# freq[i,cnt].tofile(frOut)
			plt.plot(kk*Lj, grow[i]*3600*24, 'o')

		eigVec=E.getEigenvector(i,vr,vi)

		start,end = vi.getOwnershipRange()
		if start == 0: mode[0,i,cnt] = 0; start+=1
		if end == Ny: mode[Ny,i,cnt] = 0; end -=1

		for j in range(start,end):
			mode[j,i,cnt] = 1j*vi[j] + vr[j]
			# if rank == 0:
			# 	mode[j,i,cnt].tofile(mdOut)
	print "Eigenvalues:"
	print (freq[0:5,cnt]+grow[0:5,cnt]*1j)/kx # eigenvalues

if __name__ == '__main__':
	rank = PETSc.COMM_WORLD.Get_rank()
	Print = PETSc.Sys.Print
	opts = PETSc.Options()
	nEV = opts.getInt('nev', 5)

	Ny = opts.getInt('Ny', 100)
	Ly = 350e03
	Lj = 20e03
	Hm = 500

	f0 	= 1.e-4
	bet = 0
	g0 	= 9.81
	rho = 1024
	dkk = 2e-2

	y 	= np.linspace(0,Ly, Ny+1)
	hy 	= y[1] - y[0]
	e 	= np.ones(Ny+1)

	Dy 					= sp.spdiags([-1*e, 0*e, e]/(2*hy), [-1, 0, 1], Ny+1,Ny+1).todense()
	Dy[0, 0:2] 			= [-1, 1]/hy
	Dy[Ny, Ny-1:Ny+1] 	= [-1, 1]/hy

	Eta = -0.1*np.tanh((y-Ly/2)/Lj)
	H 	= Eta + Hm
	U 	= np.ravel(-g0/f0*np.dot(Dy,Eta))
	H1 	= sp.spdiags(H, 0, Ny+1,Ny+1)
	HDy = H1*Dy
	dH  = np.ravel(np.dot(Dy,H))
	dU  = np.ravel(np.dot(Dy,U)) 

	kk = np.arange(dkk,2+dkk,dkk)/Lj
	nk = len(kk)

	grow = np.zeros([nEV,nk]) # stores imaginary part of eigenvalues
	freq = np.zeros([nEV,nk]) # stores real part of eigenvalues
	mode = np.zeros([3*Ny+1,nEV,nk], dtype=complex) # stores eigenvectors

	# # For saving the values to files
	# grOut = open('grow.dat', 'wb+')
	# frOut = open('freq.dat', 'wb+')
	# mdOut = open('mode.dat', 'wb+')

	cnt = 0
	#guess=0.21+0.09*1j
	guess = 0.21+0.045j
	for kx in kk[0:nk]: # currently just finding eigenvalues for one iteration
		k2 = kx**2

		A = construct_operator(Ny,k2,U,dU,H,dH,hy,f0,g0)
		solve_eigensystem(A, guess, nEV, cnt, freq, grow, mode)

		cnt = cnt+1

	##################

	#grOut.close(); frOut.close(); mdOut.close()

	if rank == 0: # Plotting
		ky = np.pi/Ly
		plt.ylabel('1/day')
		plt.xlabel('k')
		plt.title('Growth Rate: 1-Layer QG')
		plt.savefig('Grow1L_SW.eps', format='eps', dpi=1000)
	#plt.show()
