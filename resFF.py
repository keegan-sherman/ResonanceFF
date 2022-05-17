#! /usr/bin/env python

import functions_2 as func
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import cmath
from scipy import optimize
from scipy import integrate

mass = 1
mr = 2.2
xi = 0.5

def csqrt(z):
	arg = np.angle(z)
	norm = np.absolute(z)
	newArg = np.mod(arg + 2*math.pi, 2*math.pi) - 2*math.pi
	return np.sqrt(norm)*np.exp(1j*newArg/2)

def energy(q):
	global mass
	termOne = csqrt(pow(q,2)+pow(mass,2))
	termTwo = csqrt(pow(q,2)+pow(mass,2))
	return termOne + termTwo

def denum(q):
	global mass,mr,xi,g
	termOneNumer = xi*3*(pow(mr,2) - pow(energy(q),2))
	termOneDenum = 4*pow(g,2)*pow(mr,2)
	termTwoNumer = xi*q
	termTwoDenum = 8*cmath.pi*energy(q)
	return complex((termOneNumer/termOneDenum),-(termTwoNumer/termTwoDenum))

def denumDeriv(q):
	global mass,mr,xi,g
	tmp = ((q/csqrt(pow(q,2)+pow(mass,2))) + (q/csqrt(pow(q,2)+pow(mass,2))))
	termOneNumer = q*tmp
	termOneDenum = 8*cmath.pi*pow(energy(q),2)
	termTwoNumer = 1
	termTwoDenum = 8*cmath.pi*energy(q)
	termThreeNumer = 3*tmp*energy(q)
	termThreeDenum = 2*pow(g,2)*pow(mr,2)
	return complex(-(termThreeNumer/termThreeDenum),(termOneNumer/termOneDenum)-(termTwoNumer/termTwoDenum))

def contour_integrate(func,path,args=()):
    result=0.0
    for n in np.arange(len(path)-1):
        z0 = path[n]
        dz = path[n+1]-path[n]
        integrand_real = lambda x: np.real( func(z0+x*dz,*args)*dz )
        integrand_imag = lambda x: np.imag( func(z0+x*dz,*args)*dz )
        result_real = integrate.quad(integrand_real,0.0,1.0)[0] # keep value only
        result_imag = integrate.quad(integrand_imag,0.0,1.0)[0] # keep value only
        result += result_real + 1j*result_imag
    return result

rootsq = []
rootss = []
res = []
srData = []
c2Data = []

qrRange = np.linspace(-3,3,20)
qiRange = np.linspace(0,3,10)

gRange = [0.1,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]

for g in gRange:
    for qr in qrRange:
        for qi in qiRange:
            try:
                root = optimize.newton(denum,complex(qr,qi),denumDeriv)
                root = complex(root.real,root.imag)
                if root not in rootsq:
                    rootsq.append(root)
            except RuntimeError:
                continue
                #print("RuntimeError for: " + str(complex(qr,qi)))
    print("=======================================================================================")
    print(f"g: {g}")
    for r in rootsq:
        s = pow(energy(r),2)
        s = complex(s.real,s.imag)
        shift = min([0.01,s.imag/2])
        if s.imag < 0:
            # r_real = s.real
            # r_imag = s.imag
            C1 = (s.real-shift) + 1j*(s.imag+shift)
            C2 = (s.real+shift) + 1j*(s.imag+shift)
            C3 = (s.real+shift) + 1j*(s.imag-shift)
            C4 = (s.real-shift) + 1j*(s.imag-shift)
            C5 = C1
            C = [C1,C2,C3,C4,C5]
            # C = [C5,C4,C3,C2,C1]
            ci = contour_integrate(func.MII,C,(mass,mass,xi,mr,g))/(-2*math.pi*1j)
            # if g == 3.0:
            #     print(f"s: {s}, c: {ci}")
            srData.append(s)
            c2Data.append(-ci)
            rootsq = []
            break


Q2range = np.linspace(0,4,10)
FFdataReal = []
FFdataImag = []
tmpReal = []
tmpImag = []
FFgdataReal = []
FFgdataImag = []
termOneReal = []
termOneImag = []
termTwoReal = []
termTwoImag = []
Greal = []
Gimag = []
GrealII = []
GimagII = []
ad = []
bc = []
# print(c2Data)
for i,sr in enumerate(srData):
    g = gRange[i]
    for Q2 in Q2range:
        A = func.F(mr,Q2,g,0.1)
        f = func.f(mr,Q2)
        real,imag = func.G0(sr,Q2,sr,mass,mass,0)
        realII,imagII = func.G0II(sr,Q2,sr,mass,mass,0)
        G = complex(realII,imagII)
        if g == 3.0:
            Greal.append(real)
            Gimag.append(imag)
            GrealII.append(realII)
            GimagII.append(imagII)
            ad.append(abs(c2Data[i].real*imagII))
            bc.append(abs(c2Data[i].imag*realII))
        termOne = c2Data[i]*A
        termTwo = c2Data[i]*f*G
        if g == 3:
            termOneReal.append(termOne.real)
            termOneImag.append(termOne.imag)
            termTwoReal.append(termTwo.real)
            termTwoImag.append(termTwo.imag)
        FF = termOne + termTwo
        if Q2 == 0:
            FFgdataReal.append(FF.real)
            FFgdataImag.append(FF.imag)
        tmpReal.append(FF.real)
        tmpImag.append(FF.imag)
    FFdataReal.append(tmpReal)
    FFdataImag.append(tmpImag)
    tmpReal = []
    tmpImag = []

mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Futura'
mpl.rcParams['mathtext.it'] = 'Futura:italic'
mpl.rcParams['mathtext.bf'] = 'Futura:bold'

fig1 = plt.figure(figsize=(15,9))

plot0 = fig1.add_subplot(2,1,1)
for i,g in enumerate(gRange):
    plot0.plot(Q2range,FFdataReal[i],label=str(g),linewidth=2)
plot0.set_ylabel(r'$Re\left(f_{R\to R}(Q^{2})\right)$',size=15)
# plot0.set_xlabel(r'$Q^{2}$',size=15)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
plt.legend(title="g",loc='upper right')

plot1 = fig1.add_subplot(2,1,2)
for i,g in enumerate(gRange):
    plot1.plot(Q2range,FFdataImag[i],label=str(g),linewidth=2)
plot1.set_ylabel(r'$Im\left(f_{R\to R}(Q^{2})\right)$',size=15)
plot1.set_xlabel(r'$Q^{2}$',size=15)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)

# plt.savefig("ResFF.svg",format='svg')

fig2 = plt.figure(figsize=(15,9))

plot0g = fig2.add_subplot(2,1,1)
plot0g.plot(Q2range,termOneReal,label=r'$Re[c^{2}A_{22}]$')
plot0g.plot(Q2range,termTwoReal,label=r'$Re[c^{2}fG^{II,II}]$')
# plot0g.set_ylabel(r'$Re\left(f_{R\to R}(0)\right)$',size=15)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
plt.legend()

plot1g = fig2.add_subplot(2,1,2)
plot1g.plot(Q2range,termOneImag,label=r'$Im[c^{2}A_{22}]$')
plot1g.plot(Q2range,termTwoImag,label=r'$Im[c^{2}fG^{II,II}]$')
# plot1g.set_ylabel(r'$Im\left(f_{R\to R}(0)\right)$',size=15)
plot1g.set_xlabel(r'$Q^{2}$',size=15)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
plt.legend()

# plt.savefig("ResFF.svg",format='svg')

fig3 = plt.figure(figsize=(15,9))

plot0G = fig3.add_subplot(1,1,1)
plot0G.plot(Q2range,Greal,label=r'$Re[G]$')
plot0G.plot(Q2range,Gimag,label=r'$Im[G]$')
plot0G.plot(Q2range,GrealII,label=r'$Re[G^{II,II}]$')
plot0G.plot(Q2range,GimagII,label=r'$Im[G^{II,II}]$')
plot0G.set_xlabel(r'$Q^{2}$',size=15)
plt.title(r'$c^{2}=$'+str(c2Data[6]),size=15)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
plt.legend()

# plt.savefig("G.svg",format='svg')

fig4 = plt.figure(figsize=(15,9))

plot0comp = fig4.add_subplot(1,1,1)
plot0comp.plot(Q2range,ad,label=r'$|Re[c^{2}]*Im[G^{II,II}]|$')
plot0comp.plot(Q2range,bc,label=r'$|Im[c^{2}]*Re[G^{II,II}]|$')
plot0comp.set_xlabel(r'$Q^{2}$',size=15)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
plt.legend()

# plt.savefig("comp.svg",format='svg')

plt.show()