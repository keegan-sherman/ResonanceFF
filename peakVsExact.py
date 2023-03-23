#! /usr/bin/env python

import functions_2 as func
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import cmath
import sys
import argparse as ap
from scipy import optimize
from scipy import integrate
import matplotlib.font_manager as font_manager
from matplotlib.widgets import Slider

mass = 1
mr = 2.2
xi = 0.5

sth = pow(2*mass,2)

parser = ap.ArgumentParser()
parser.add_argument('-g',nargs='+',default=[0.1],type=float)
parser.add_argument('-s',default='none',type=str)
parser.add_argument('-p',action='store_true')

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

# def denumDeriv(q):
# 	global mass,mr,xi,g
# 	tmp = ((q/csqrt(pow(q,2)+pow(mass,2))) + (q/csqrt(pow(q,2)+pow(mass,2))))
# 	termOneNumer = q*tmp
# 	termOneDenum = 8*cmath.pi*pow(energy(q),2)
# 	termTwoNumer = 1
# 	termTwoDenum = 8*cmath.pi*energy(q)
# 	termThreeNumer = 3*tmp*energy(q)
# 	termThreeDenum = 2*pow(g,2)*pow(mr,2)
# 	return complex(-(termThreeNumer/termThreeDenum),(termOneNumer/termOneDenum)-(termTwoNumer/termTwoDenum))

def denumDeriv(q):
    global mass,mr,xi,g
    tmp = csqrt(pow(q,2)+pow(mass,2))
    termOne = (-6*xi*q)/(pow(g,2)*pow(mr,2))
    termTwo = (xi*pow(q,2))/(16*cmath.pi*pow(tmp,3))
    termThree = (-xi)/(16*cmath.pi*tmp)
    return complex(termOne,(termTwo+termThree))

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
qiRange = np.linspace(0,-3.0,10)
for g in gRange:
    for qr in qrRange:
        for qi in qiRange:
            try:
                root = optimize.newton(denum,complex(qr,qi),fprime=denumDeriv)
                if root not in rootsq:
                    rootsq.append(root)
            except (RuntimeError, OverflowError):
                continue

    for r in rootsq:
            s = pow(energy(r),2)
            s = complex(s.real,s.imag)
            shift = min([0.01,s.imag/2])
            if s.imag < 0:
                C1 = (s.real-shift) + 1j*(s.imag+shift)
                C2 = (s.real+shift) + 1j*(s.imag+shift)
                C3 = (s.real+shift) + 1j*(s.imag-shift)
                C4 = (s.real-shift) + 1j*(s.imag-shift)
                C5 = C1
                C = [C1,C2,C3,C4,C5]
                ci = contour_integrate(func.MII,C,(mass,mass,xi,mr,g))/(-2*math.pi*1j)
                srData.append(s)
                c2Data.append(-ci)
                rootsq = []
                break


if args['p']:
    print("sr: c^2")
    for i,j in enumerate(srData):
        print(f"{j}, {(np.angle(j)*180/math.pi)+360}: {c2Data[i]}")
    sys.exit(0)
    

print(f"sr: {srData}")
print(f"c: {c2Data}")

#endregion

# Get form factors from pole and peak and calculate ratio
#region
Q2start = 0
Q2end = 4.5
Q2size = 500
Q2step = (Q2end-Q2start)/Q2size
efSize = 1000
Q2range = np.linspace(Q2start,Q2end,Q2size)
efRange = np.linspace(1.0,3.0,efSize)
sfRange = np.power(efRange,2)
fdata = np.array([func.f(mr,Q2) for Q2 in Q2range])
GpeakData = np.zeros((Q2size),dtype=complex)
GexactData = np.zeros((Q2size),dtype=complex)
peak_sData = {}
peak_indexData = {}
Mdata = {
    g: np.array([abs(func.M(i, mass, mass, xi, mr, g)) for i in sfRange])
    for g in gRange
}

FFexact = []
FFpeak = []
ratio = []

error = {}
WidthMassRatio = []

for i,sr in enumerate(srData):
    g = gRange[i]

    peak_index = Mdata[g].argmax()
    peak_indexData[g] = peak_index
    peak_energy = efRange[peak_index]
    peak_s = pow(peak_energy,2)
    peak_sData[g] = peak_s
    print(f"g:{g}, peak:{peak_s}")

    Adata = np.array([func.F(mr,Q2,g,0.1) for Q2 in Q2range])
    for j,Q2 in enumerate(Q2range):
        realPeak,imagPeak = func.G0(peak_s,Q2,peak_s,mass,mass,0)
        GpeakData[j] = complex(realPeak,imagPeak)
        realExact,imagExact = func.G0II(sr,Q2,sr,mass,mass,0)
        GexactData[j] = complex(realExact,imagExact)

    FFexact = abs(c2Data[i]*(Adata+fdata*GexactData))
    FFpeak = abs(cgSq(g,mr,xi)*(Adata+fdata*GpeakData))
    # print(f"A: {Adata[0]}")
    # print(f"f: {fdata[0]}")
    # print(f"G peak: {GpeakData[0]}, exact: {GexactData[0]}")
    # print(f"FF peak: {FFpeak[0]}, exact: {FFexact[0]}")
    # print()
    r = FFpeak/FFexact
    ratio.append(r)
    # error.append(abs((r[0]-1)*100))

    for k,tmp in enumerate(r):
        # if k == 0:
        #     print(f"r: {tmp}")
        if not k%5:
            if k in error:
                error[k].append(abs((tmp-1)*100))
            else:
                error[k] = [abs((tmp-1)*100)] 

    er = csqrt(sr)
    massr = er.real 
    widthr = -2*er.imag
    WidthMassRatio.append(widthr/massr)

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
    plot1_0.plot(Q2range,ratio[i],label=str(g),linewidth=3,color=colors[g])
plot1_0.set_ylabel(r"$\left| f_{R\to R, peak}(Q^{2})/f_{R\to R, exact}(Q^{2}) \right|$",size=20)
plot1_0.set_xlabel(r"$Q^{2}$",size=20)
plot1_0.set_ylim(0.8,1.5)
ratioLine = plot1_0.axvline(0.0,color='black')
# plt.axvline(4.0,color='black')
plt.xticks(fontname="Futura",fontsize=20)
plt.yticks(fontname="Futura",fontsize=20)
plt.legend(title="g",loc='upper right',title_fontsize=20,fontsize=15)

plt.savefig("ratioPlot_Notes1.svg",format='svg')

fig3 = plt.figure(figsize=(11,6.5))

ax = plt.subplot2grid((6,7),(0,0),colspan=6,rowspan=4)
ax.set_ylabel(r'$|\mathcal{M}|$',size=20)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlim(2.0, 9.0)
plt.ylim(-7,300)
plt.axvline(x=sth, color='black')
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,position=(0,0.25))

for g in gRange:
	plt.plot(sfRange, Mdata[g], label=f"g={str(g)}", color=colors[g])
	# plt.scatter(peak_sData[g],Mdata[g][peak_indexData[g]],color=colors[g])

# plt.savefig("M3.svg",format='svg')

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

plt.xlim(2.0, 9.0)
plt.ylim(-1.0, 0.03)

ax.set_ylabel(r'Im($E_{cm}^{2}$) (GeV$^{2}$)', fontname="Futura", size=20)
ax.set_xlabel(r'Re($E_{cm}^{2}$) (GeV$^{2}$)',fontname="Futura",size=20)
ax.xaxis.set_label_coords(82.0/72.0,90.0/72.0)

# plt.savefig("PoleTalk_1.svg",format='svg')
# plt.savefig("PeaksNPoles_0P1-3P0.svg",format='svg')

fig4 = plt.figure(figsize=(15,9))

plot1_4 = fig4.add_subplot(1,1,1)
plt.subplots_adjust(left=0.15, bottom=0.15)
pl1_4, = plot1_4.plot(WidthMassRatio,error[0],linewidth=3)
plot1_4.set_ylabel(r"$\sigma(\%)$",size=20)
plot1_4.set_xlabel(r"$\Gamma/m_{r}$",size=20)
plt.xticks(fontname="Futura",fontsize=20)
plt.yticks(fontname="Futura",fontsize=20)
plot1_4.set_title(r"$Q^{2}=0$",fontsize=20)

# axcolor = 'lightgoldenrodyellow'
# axQ2Val = plt.axes([0.15,0.05,0.75,0.03],facecolor=axcolor)

# sQ2Val = Slider(axQ2Val, r'$Q^{2}$', 0.0, Q2size, valinit=0.0, valstep=5)

# def updateQ2(Q2Val):
#     Q2index = sQ2Val.val
#     pl1_4.set_ydata(error[Q2index])
#     plot1_4.set_ylim(0,max(error[Q2index])+0.05)
#     plot1_4.set_title(r"$Q^{2}=$"+str(Q2index*Q2step))
#     # plot1_0.axvline(Q2index*Q2step,color='black')
#     ratioLine.set_xdata(Q2index*Q2step)
#     fig4.canvas.draw_idle()
#     fig1.canvas.draw_idle()

# sQ2Val.on_changed(updateQ2)

# plt.savefig("ErrorNotes.svg",format='svg')

plt.show()

#endregion

