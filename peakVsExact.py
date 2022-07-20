#! /usr/bin/env python

import functions_2 as func
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import cmath
import argparse as ap
from scipy import optimize
from scipy import integrate
import matplotlib.font_manager as font_manager

mass = 1
mr = 2.2
xi = 0.5

sth = pow(2*mass,2)

parser = ap.ArgumentParser()
parser.add_argument('-g',nargs='+',default=[0.1],type=float)
parser.add_argument('-s',default='none',type=str)

args = vars(parser.parse_args())

gRange = args['g']


# Define functions 
#region

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

def cgSq(g,mr,xi):
    return 4*pow(g,2)*pow(mr,2)/(3*xi)

#endregion

# Find poles and residues
#region

rootsq = []
srData = []
c2Data = []

qrRange = np.linspace(-4.5,4.5,20)
qiRange = np.linspace(0,3,10)
for g in gRange:
    for qr in qrRange:
            for qi in qiRange:
                try:
                    root = optimize.newton(denum,complex(qr,qi),denumDeriv)
                    if root not in rootsq:
                        rootsq.append(root)
                except RuntimeError:
                    continue

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

print(f"sr: {srData}")
print(f"c: {c2Data}")

#endregion

# Get form factors from pole and peak
#region
Q2Size = 500
efSize = 1000
Q2range = np.linspace(0,4.5,Q2Size)
efRange = np.linspace(1.0,3.0,efSize)
sfRange = np.power(efRange,2)
Adata = np.array([func.F(mr,Q2,g,0.1) for Q2 in Q2range])
fdata = np.array([func.f(mr,Q2) for Q2 in Q2range])
GpeakData = np.zeros((Q2Size),dtype=complex)
GexactData = np.zeros((Q2Size),dtype=complex)
peak_sData = {}
peak_indexData = {}
# Mdata = np.zeros((efSize),dtype=complex)
Mdata = {}
for g in gRange:
    Mdata[g] = np.array([abs(func.M(i,mass,mass,xi,mr,g)) for i in sfRange]) 

# Mdata = np.zeros((len(gRange),efSize),dtype=complex)

FFexact = []
FFpeak = []
ratio = []

print(f"c2: {c2Data}")
print(f"sr: {srData}")

for i,sr in enumerate(srData):
    g = gRange[i]
    # for j,ef in enumerate(efRange):
    #     sf = pow(ef,2)
    #     Mdata[i][j] = func.M(sf,mass,mass,xi,mr,g)

    peak_index = Mdata[g].argmax()
    peak_indexData[g] = peak_index
    peak_energy = efRange[peak_index]
    peak_s = pow(peak_energy,2)
    peak_sData[g] = peak_s
    print(f"g:{g}, peak:{peak_s}")

    for j,Q2 in enumerate(Q2range):
        realPeak,imagPeak = func.G0(peak_s,Q2,peak_s,mass,mass,0)
        # realPeak,imagPeak = func.G0II(pow(mr,2),Q2,pow(mr,2),mass,mass,0)
        GpeakData[j] = complex(realPeak,imagPeak)
        realExact,imagExact = func.G0II(sr,Q2,sr,mass,mass,0)
        GexactData[j] = complex(realExact,imagExact)

    FFexact = abs(c2Data[i]*(Adata+fdata*GexactData))
    FFpeak = abs(cgSq(g,mr,xi)*(Adata+fdata*GpeakData))
    ratio.append(FFpeak/FFexact)
    # print(f"Exact: {FFexact}, Peak: {FFpeak}, ratio: {ratio[i]}")

#endregion

# Plots
#region

mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Futura'
mpl.rcParams['mathtext.it'] = 'Futura:italic'
mpl.rcParams['mathtext.bf'] = 'Futura:bold'

fig1  = plt.figure(figsize=(15,9))
colors = {0.1:"pink", 0.5:"red", 1.0:"coral", 1.5:"orange", 2.0:"green", 2.5:"cyan", 3.0:"blue", 3.5:"purple", 4.0:"brown", 4.5:"gray", 5.0:"black" }

plot1_0 = fig1.add_subplot(1,1,1)
for i,g in enumerate(gRange):
    plot1_0.plot(Q2range,ratio[i],label=str(g),linewidth=2,color=colors[g])
    # plot1_0.plot(Q2range,np.real(FFexact),label="exact",linewidth=2)
    # plot1_0.plot(Q2range,np.real(FFpeak),label="peak",linewidth=2)
plot1_0.set_ylabel(r"$\left| f_{R\to R, peak}(Q^{2})/f_{R\to R, exact}(Q^{2}) \right|$",size=15)
plot1_0.set_xlabel(r"$Q^{2}$",size=15)
# plt.axvline(4.0,color='black')
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
plt.legend(title="g",loc='upper right')

# plot1 = fig1.add_subplot(2,1,2)
# for i,g in enumerate(gRange):
#     plot1.plot(Q2range,np.imag(ratio[i]),label=str(g),linewidth=2)
#     # plot1.plot(Q2range,np.imag(FFexact),label="exact",linewidth=2)
#     # plot1.plot(Q2range,np.imag(FFpeak),label="peak",linewidth=2)
# plot1.set_ylabel(r"$Im\left[ f_{R\to R, peak}(Q^{2})/f_{R\to R, exact}(Q^{2}) \right]$",size=15)
# plot1.set_xlabel(r"Q^{2}",size=15)
# plt.xticks(fontname="Futura",fontsize=15)
# plt.yticks(fontname="Futura",fontsize=15)

# plt.savefig("PeakVsExact_Sing.svg",format='svg')

# fig2 = plt.figure(figsize=(15,9))

# plot2_0 = fig2.add_subplot(1,1,1)
# plot2_0.plot(Q2range,abs(GpeakData),label="G Peak",linewidth=2)
# plot2_0.plot(Q2range,abs(GexactData),label="G Exact",linewidth=2)
# plot2_0.plot(Q2range,Adata,label="A",linewidth=2)
# plot2_0.set_xlabel(r"$Q^{2}$")
# plt.xticks(fontname="Futura",fontsize=15)
# plt.yticks(fontname="Futura",fontsize=15)
# plt.legend(loc='upper right')

# plot2_1 = fig2.add_subplot(2,1,2)
# plot2_1.plot(Q2range,GpeakData.imag,label="Peak",linewidth=2)
# plot2_1.plot(Q2range,GexactData.imag,label="Exact",linewidth=2)
# plot2_1.set_xlabel(r"$Q^{2}$")
# plt.xticks(fontname="Futura",fontsize=15)
# plt.yticks(fontname="Futura",fontsize=15)

fig3 = plt.figure(figsize=(11,6.5))

ax = plt.subplot2grid((6,7),(0,0),colspan=6,rowspan=4)
ax.set_ylabel(r'$|\mathcal{M}|$',size=20)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlim(2.0, 9.0)
plt.ylim(-7,400)
plt.axvline(x=sth, color='black')
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,position=(0,0.25))

for g in gRange:
    # plt.plot(sfRange1,data1[i],color=colors[g],alpha=0.25)
    plt.plot(sfRange,Mdata[g],label="g="+str(g),color=colors[g])
    plt.scatter(peak_sData[g],Mdata[g][peak_indexData[g]],color=colors[g])

font = font_manager.FontProperties(family="Futura",size=12)
plt.legend(prop=font,loc='upper right')

ax = plt.subplot2grid((7,7),(5,0),colspan=6,rowspan=2)
ax.xaxis.set_ticks_position('top')
for tick in ax.get_xticklabels():
	tick.set_verticalalignment('baseline')
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

for i,g in enumerate(gRange):
	plt.scatter(srData[i].real,srData[i].imag,color=colors[g])

plt.xlim(0.9, 9.0)
plt.ylim(-1.0, 0.03)

ax.set_ylabel(r'Im(s) (GeV$^{2}$)', fontname="Futura", size=20)
ax.set_xlabel(r'Re(s) (GeV$^{2}$)',fontname="Futura",size=20)
ax.xaxis.set_label_coords(82.0/72.0,90.0/72.0)


plt.show()

#endregion

