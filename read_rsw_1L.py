import numpy as np
import matplotlib.pyplot as plt

# read from files
# data array contains: Ly,Lj,Hm,Ny,f0,bet,g0,rho,dkk,nEV
data = np.fromfile('storage/InputData')
grow = np.fromfile('storage/grow')
freq = np.fromfile('storage/freq')
mode = np.fromfile('storage/mode',dtype=np.complex128)
kk = np.fromfile('storage/kk')
evalsArr = np.fromfile('storage/evals')

nk = len(kk)
nEV = data[9]
Ny = data[3]
grow = grow.reshape([nEV,nk])
freq = freq.reshape([nEV,nk])
mode = mode.reshape([3*Ny+1,nEV,nk])

for i in range(nk):
    for j in range(int(evalsArr[i])):
        plt.plot(np.arange(0,nk),grow[j,:]*3600*24, 'o')

plt.ylabel('1/day')
plt.xlabel('k/dk')
plt.title('Growth Rate: 1-Layer SW2')
plt.savefig('Grow1L_SW.eps', format='eps', dpi=1000)
plt.show()
