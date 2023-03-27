#! /Users/keegan/anaconda3/ python

import math
import numpy as np
import tri_lib
import matplotlib as mpl
import matplotlib.pyplot as plt


m1 = 1
m2 = 1
eth = m1+m2
ei = 2.05
Pi_3Vec = [0,0,0]
Pf_3Vec = [0,1,1]
efRange = np.linspace(1.9,2.5,1000)
efRangeScaled = np.divide(efRange,eth)
epsilon = 0

PWaveScalarG = [[[],[],[]],[[],[],[]],[[],[],[]]]
I23Data = []
I24Data = []

Pi = np.array([np.sqrt(pow(ei,2)+np.dot(Pi_3Vec,Pi_3Vec)),Pi_3Vec[0],Pi_3Vec[1],Pi_3Vec[2]])

print("Starting main loop")
for count,ef in enumerate(efRange):
    Pf = np.array([np.sqrt(pow(ef,2)+np.dot(Pf_3Vec,Pf_3Vec)),Pf_3Vec[0],Pf_3Vec[1],Pf_3Vec[2]])
    si = tri_lib.dot(Pi,Pi)
    sf = tri_lib.dot(Pf,Pf)
    q = np.subtract(Pf,Pi)
    Q2 = -1*tri_lib.dot(q,q)

    tmp = tri_lib.G_PwaveScalar(Pi,Pf,m1,m2,epsilon)
    I23Data.append(tri_lib.I23(si,Q2,sf,m1,m2,epsilon))
    I24Data.append(tri_lib.I24(si,Q2,sf,m1,m2,epsilon))
    if count%100==0: print(f"On iteration {count}")
    for i in range(len(PWaveScalarG)):
        for j in range(len(PWaveScalarG[i])):
            PWaveScalarG[i][j].append(tmp[i][j])

# print(PWaveScalarG)
PWaveScalarG = np.array(PWaveScalarG)
I23Data = np.array(I23Data)
I24Data = np.array(I24Data)

# Plotting
#region

mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Futura'
mpl.rcParams['mathtext.it'] = 'Futura:italic'
mpl.rcParams['mathtext.bf'] = 'Futura:bold'

fig1 = plt.figure(figsize=(15,9))

plot_n1n1 = fig1.add_subplot(3,3,1)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot_n1n1.plot(efRangeScaled,np.real(PWaveScalarG[0][0]),color="red")
plot_n1n1.plot(efRangeScaled,np.imag(PWaveScalarG[0][0]),color="blue")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.title(r"$m_{\ell'}=-1$ $m_{\ell}=-1$")

plot_n10 = fig1.add_subplot(3,3,2)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot_n10.plot(efRangeScaled,np.real(PWaveScalarG[0][1]),color="red")
plot_n10.plot(efRangeScaled,np.imag(PWaveScalarG[0][1]),color="blue")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.title(r"$m_{\ell'}=-1$ $m_{\ell}=0$")

plot_n11 = fig1.add_subplot(3,3,3)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot_n11.plot(efRangeScaled,np.real(PWaveScalarG[0][2]),color="red")
plot_n11.plot(efRangeScaled,np.imag(PWaveScalarG[0][2]),color="blue")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.title(r"$m_{\ell'}=-1$ $m_{\ell}=1$")

plot_0n1 = fig1.add_subplot(3,3,4)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot_0n1.plot(efRangeScaled,np.real(PWaveScalarG[1][0]),color="red")
plot_0n1.plot(efRangeScaled,np.imag(PWaveScalarG[1][0]),color="blue")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.title(r"$m_{\ell'}=0$ $m_{\ell}=-1$")

plot_00 = fig1.add_subplot(3,3,5)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot_00.plot(efRangeScaled,np.real(PWaveScalarG[1][1]),color="red")
plot_00.plot(efRangeScaled,np.imag(PWaveScalarG[1][1]),color="blue")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.title(r"$m_{\ell'}=0$ $m_{\ell}=0$")

plot_01 = fig1.add_subplot(3,3,6)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot_01.plot(efRangeScaled,np.real(PWaveScalarG[1][2]),color="red")
plot_01.plot(efRangeScaled,np.imag(PWaveScalarG[1][2]),color="blue")
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.title(r"$m_{\ell'}=0$ $m_{\ell}=1$")

plot_1n1 = fig1.add_subplot(3,3,7)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot_1n1.plot(efRangeScaled,np.real(PWaveScalarG[2][0]),color="red")
plot_1n1.plot(efRangeScaled,np.imag(PWaveScalarG[2][0]),color="blue")
plt.title(r"$m_{\ell'}=1$ $m_{\ell}=-1$")

plot_10 = fig1.add_subplot(3,3,8)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot_10.plot(efRangeScaled,np.real(PWaveScalarG[2][1]),color="red")
plot_10.plot(efRangeScaled,np.imag(PWaveScalarG[2][1]),color="blue")
plt.title(r"$m_{\ell'}=1$ $m_{\ell}=0$")

plot_11 = fig1.add_subplot(3,3,9)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot_11.plot(efRangeScaled,np.real(PWaveScalarG[2][2]),color="red")
plot_11.plot(efRangeScaled,np.imag(PWaveScalarG[2][2]),color="blue")
plt.title(r"$m_{\ell'}=1$ $m_{\ell}=1$")

# plt.savefig("PWaveScalarG.svg",format='svg')
 
fig2 = plt.figure(figsize=(15,9))

plot21 = fig2.add_subplot(2,1,1)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot21.plot(efRangeScaled,np.real(I23Data))

plot22 = fig2.add_subplot(2,1,2)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot22.plot(efRangeScaled,np.imag(I23Data))

plt.savefig("I23.svg",format='svg')

fig3 = plt.figure(figsize=(15,9))

plot31 = fig3.add_subplot(2,1,1)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot31.plot(efRangeScaled,np.real(I24Data))

plot32 = fig3.add_subplot(2,1,2)
plt.axhline(0,color='black')
plt.axvline(1.0,color='black')
plot32.plot(efRangeScaled,np.imag(I24Data))

plt.savefig("I24.svg",format='svg')

plt.show()

#endregion