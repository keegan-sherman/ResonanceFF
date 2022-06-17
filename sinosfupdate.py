# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import pylab
import numpy as np
import mpmath
import physfunc as pf
import math
import cmath
from Pole_Finder_for_BW_class import BWPoleFinder
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

# warnings.filterwarnings("ignore")

Bcon = .1
m1 = 1
m2 = 1
g = 3
mr = 2.2
xi = 1/2
ascale = .0001
# disp = 0
disp = .110117
pla = .110117
# disp = 0.47915701951257783
# e = 0

ei = 1.05*(m1+m2)
# si = 0
# Q2 = 1.5
eps = 0
efrange = np.linspace(.05+disp*1j,3+disp*1j,1000)
Pz = 2*math.pi/15
Q0 = 2

def csqrt(z):
    arg = np.angle(z)
    norm = np.absolute(z)
    newarg = np.mod(arg + 2*math.pi, 2*math.pi)
    return np.sqrt(norm)*np.exp(1j*newarg/2)

def arctangent(z):
    return np.arctan(z)

#equal masses qstar
# def qStar(s):
#     m1 = 1
#     m2 = 1
#     return 1/2 * csqrt(s - 4)

#not equal masses qstar
def qStar(s):
    return 1/2 * csqrt((s - 2*((m1**2)+(m2**2)) + ((((m2**2)-(m1**2))**2)/s)))

def rho(s):
    top = xi*qStar(s)
    bottom = 8*math.pi*cmath.sqrt(s)
    return (top/bottom)

#this contains a psuedo point for plotting
def rhoforplot(s):
    r = rho(s)
    if r > .05:
        return .05
    elif r == math.inf:
        return .5
    return r

def gamma(s):
    f = pow(g,2)/(6.0*cmath.pi)
    return f*pow(mr,2)*(qStar(s))/s

def tanBW(s):
    bottom = pow(mr,2)-s
    return np.sqrt(s)*gamma(s) / bottom

#this contains a psuedo point for plotting
def tanBWforplot(s):
    t = tanBW(s)
    if t > 3:
        return 3
    elif t == math.inf:
        return 3
    return t

def Kmatrix(s):
    top = 4*pow(g,2)*pow(mr,2)
    bottom = 3*xi*(pow(mr,2)-s)
    return top/bottom

#This is the real component M on the first sheet
def MI_real(s):
    a = np.real(1/((1/Kmatrix(s)) - (1j*rho(s))))
    b = np.imag(1/((1/Kmatrix(s)) - (1j*rho(s))))
    return a

#This is the imaginary component M on the first sheet
def MI_imag(s):
    a = np.real(1/((1/Kmatrix(s)) - (1j*rho(s))))
    b = np.imag(1/((1/Kmatrix(s)) - (1j*rho(s))))
    return b

#This is the real component M on the second sheet
def MII_real(s):
    a = np.real(1/((1/Kmatrix(s)) + (1j*rho(s))))
    b = np.imag(1/((1/Kmatrix(s)) + (1j*rho(s))))
    return a

#This is the imaginary component M on the second sheet
def MII_imag(s):
    a = np.real(1/((1/Kmatrix(s)) + (1j*rho(s))))
    b = np.imag(1/((1/Kmatrix(s)) + (1j*rho(s))))
    return b

def A(x,m1,m2,Q2,si,sf,eps):
    sth = pow((m1+m2),2)
    num1 = (pow(m2,2)-pow(m1,2)-(x*(Q2+sf+si)))
    frac1 = num1/si
    return 1 + frac1

def B(x,m1,m2,Q2,si,sf,eps):
    sth = pow((m1+m2),2)
    num2 = (pow(m2,2)-((x*((pow(m2,2))-pow(m1,2))))-(x*(1-x)*sf))
    frac2 = num2/si
    return (-4) * frac2

def yplus(x,m1,m2,Q2,si,sf,eps):
    sqrt = csqrt(A(x,m1,m2,Q2,si,sf,eps)**2+B(x,m1,m2,Q2,si,sf,eps)+(1j*eps))
    return .5 * (A(x,m1,m2,Q2,si,sf,eps)+sqrt)

def yminus(x,m1,m2,Q2,si,sf,eps):
    sqrt = csqrt(A(x,m1,m2,Q2,si,sf,eps)**2+B(x,m1,m2,Q2,si,sf,eps)+(1j*eps))
    return .5 * (A(x,m1,m2,Q2,si,sf,eps)-sqrt)

def Lplus(x,m1,m2,Q2,si,sf,eps):
    ins = (1-x-yplus(x,m1,m2,Q2,si,sf,eps))/(yplus(x,m1,m2,Q2,si,sf,eps))
    abs = np.abs(ins)
    log = cmath.log(abs)
    Rey = np.real(yplus(x,m1,m2,Q2,si,sf,eps))
    Imy = np.imag(yplus(x,m1,m2,Q2,si,sf,eps))

    if (Imy == 0 and 1-x-Rey >= 0):
        iarc1 = 1j*cmath.pi/2
    if (Imy == 0 and Rey >= 0):
        iarc2 = 1j*cmath.pi/2
    if (Imy == 0 and 1-x-Rey < 0):
        iarc1 = 1j*-1*cmath.pi/2
    if (Imy == 0 and Rey < 0):
        iarc2 = 1j*-1*cmath.pi/2

    if (Imy != 0):
        arc1 = np.arctan((1-x-Rey)/((Imy)+eps))
        arc2 = np.arctan((Rey)/((Imy)+eps))
        iarc1 = 1j * arc1
        iarc2 = 1j * arc2

    return (log + iarc1 + iarc2)

def Lminus(x,m1,m2,Q2,si,sf,eps):
    abs2 = np.abs((1-x-yminus(x,m1,m2,Q2,si,sf,eps))/(yminus(x,m1,m2,Q2,si,sf,eps)))
    log2 = np.log(abs2)
    Reyb = np.real(yminus(x,m1,m2,Q2,si,sf,eps))
    Imyb = np.imag(yminus(x,m1,m2,Q2,si,sf,eps))

    if (Imyb == 0 and 1-x-Reyb >= 0):
        iarc12 = 1j*-1*cmath.pi/2
    if (Imyb == 0 and Reyb >= 0):
        iarc22 = 1j*-1*cmath.pi/2
    if (Imyb == 0 and 1-x-Reyb < 0):
        iarc12 = 1j*cmath.pi/2
    if (Imyb == 0 and Reyb < 0):
        iarc22 = 1j*cmath.pi/2

    if (Imyb != 0):
        arc12 = np.arctan((1-x-Reyb)/((Imyb)-eps))
        arc22 = np.arctan((Reyb)/((Imyb)-eps))
        iarc12 = 1j * arc12
        iarc22 = 1j * arc22

    return log2 + iarc12 + iarc22

def realF(x,si,sf,Q2):
    sth = pow((m1+m2),2)
    fpi = 4*cmath.pi
    fpisq = pow(fpi,2)
    lead = 1 / (fpisq*si)
    numf = (Lplus(x,m1,m2,Q2,si,sf,eps)-Lminus(x,m1,m2,Q2,si,sf,eps))
    denf = (yplus(x,m1,m2,Q2,si,sf,eps)-yminus(x,m1,m2,Q2,si,sf,eps))
    fracf = numf/denf
    return np.real(lead * fracf)

def imagF(x,si,sf,Q2):
    sth = pow((m1+m2),2)
    fpi = 4*cmath.pi
    fpisq = pow(fpi,2)
    lead = 1 / (fpisq*si)
    numf = (Lplus(x,m1,m2,Q2,si,sf,eps)-Lminus(x,m1,m2,Q2,si,sf,eps))
    denf = (yplus(x,m1,m2,Q2,si,sf,eps)-yminus(x,m1,m2,Q2,si,sf,eps))
    fracf = numf/denf
    return np.imag(lead * fracf)

def realF11(x,si,sf,Q2):
    return x * realF(x,si,sf,Q2)

def imagF11(x,si,sf,Q2):
    return x * imagF(x,si,sf,Q2)

def realF12(x,si,sf,Q2):
    sth = pow((m1+m2),2)
    fpi = 4*cmath.pi
    fpisq = pow(fpi,2)
    lead = 1 / (fpisq*si)
    numf = ((yplus(x,m1,m2,Q2,si,sf,eps)*Lplus(x,m1,m2,Q2,si,sf,eps))-(yminus(x,m1,m2,Q2,si,sf,eps)*Lminus(x,m1,m2,Q2,si,sf,eps)))
    denf = (yplus(x,m1,m2,Q2,si,sf,eps)-yminus(x,m1,m2,Q2,si,sf,eps))
    fracf = numf/denf
    return np.real(lead * fracf)

def imagF12(x,si,sf,Q2):
    sth = pow((m1+m2),2)
    fpi = 4*cmath.pi
    fpisq = pow(fpi,2)
    lead = 1 / (fpisq*si)
    numf = ((yplus(x,m1,m2,Q2,si,sf,eps)*Lplus(x,m1,m2,Q2,si,sf,eps))-(yminus(x,m1,m2,Q2,si,sf,eps)*Lminus(x,m1,m2,Q2,si,sf,eps)))
    denf = (yplus(x,m1,m2,Q2,si,sf,eps)-yminus(x,m1,m2,Q2,si,sf,eps))
    fracf = numf/denf
    return np.imag(lead * fracf)


#####################
#this is the TRUE M
def MI(sf):
    return MI_real(sf) +(1j*(MI_imag(sf)))

def MII(sf):
    return MII_real(sf) + (1j*(MII_imag(sf)))

def IaFr(si,sf,Q2):
    # crit = np.real(pf.crits(m1,m2,Q2,si,sf,0))
    # IaR,error = scipy.integrate.quad(realF,0,1,args=(si,sf,Q2),points=crit)
    IaR,error = scipy.integrate.quad(realF,0,1,args=(si,sf,Q2))
    return (IaR)

def IaFi(si,sf,Q2):
    # crit = np.real(pf.crits(m1,m2,Q2,si,sf,0))
    # IaI,error2 = scipy.integrate.quad(imagF,0,1,args=(si,sf,Q2),points=crit)
    IaI,error2 = scipy.integrate.quad(imagF,0,1,args=(si,sf,Q2))
    return (IaI)

def IaFr1(si,sf,Q2):
    # crit = np.real(pf.crits(m1,m2,Q2,si,sf,0))
    # IaR,error = scipy.integrate.quad(realF11,0,1,args=(si,sf,Q2),points=crit)
    IaR,error = scipy.integrate.quad(realF11,0,1,args=(si,sf,Q2))
    return (IaR)

def IaFi1(si,sf,Q2):
    # crit = np.real(pf.crits(m1,m2,Q2,si,sf,0))
    # IaR,error = scipy.integrate.quad(imagF11,0,1,args=(si,sf,Q2),points=crit)
    IaR,error = scipy.integrate.quad(imagF11,0,1,args=(si,sf,Q2))
    return (IaR)

def IaFr2(si,sf,Q2):
    # crit = np.real(pf.crits(m1,m2,Q2,si,sf,0))
    # IaR,error = scipy.integrate.quad(realF12,0,1,args=(si,sf,Q2),points=crit)
    IaR,error = scipy.integrate.quad(realF12,0,1,args=(si,sf,Q2))
    return (IaR)

def IaFi2(si,sf,Q2):
    # crit = np.real(pf.crits(m1,m2,Q2,si,sf,0))
    # IaR,error = scipy.integrate.quad(imagF12,0,1,args=(si,sf,Q2),points=crit)
    IaR,error = scipy.integrate.quad(imagF12,0,1,args=(si,sf,Q2))
    return (IaR)

#scalar G
def GI(si,sf,Q2):
    # b = np.subtract(Pf,Pi)
    # s = pf.dotprod(b,b)
    # Q2=-1*(s)
    return (IaFr(si,sf,Q2) + 1j*IaFi(si,sf,Q2))

def GII(si,sf,Q2):
    # b = np.subtract(Pf,Pi)
    # s = pf.dotprod(b,b)
    # Q2=-1*(s)
    return (IaFr(si,sf,Q2) - 1j*IaFi(si,sf,Q2))

#combination of vector GS
def ComboG_a(si,sf,Q2):
    # b = np.subtract(Pf,Pi)
    # s = pf.dotprod(b,b)
    # Q2=-1*(s)
    return (IaFr1(si,sf,Q2) + 1j*IaFi1(si,sf,Q2))

def ComboG_b(si,sf,Q2):
    # b = np.subtract(Pf,Pi)
    # s = pf.dotprod(b,b)
    # Q2=-1*(s)
    return (IaFr2(si,sf,Q2) + 1j*IaFi2(si,sf,Q2))

def ComboG_aII(si,sf,Q2):
    # b = np.subtract(Pf,Pi)
    # s = pf.dotprod(b,b)
    # Q2=-1*(s)
    return (IaFr1(si,sf,Q2) - 1j*IaFi1(si,sf,Q2))

def ComboG_bII(si,sf,Q2):
    # b = np.subtract(Pf,Pi)
    # s = pf.dotprod(b,b)
    # Q2=-1*(s)
    return (IaFr2(si,sf,Q2) - 1j*IaFi2(si,sf,Q2))

def Gvec(Pi,Pf,Q2):
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)

    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)

    Piterm = Pi*(GI(si,sf,Q2)-2*ComboG_b(si,sf,Q2))
    Pfterm = Pf*(GI(si,sf,Q2)-2*ComboG_a(si,sf,Q2))
    out = Piterm + Pfterm
    return out

def GvecII(Pi,Pf,Q2):
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)

    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)

    Piterm = Pi*(GII(si,sf,Q2)-2*ComboG_bII(si,sf,Q2))
    Pfterm = Pf*(GII(si,sf,Q2)-2*ComboG_aII(si,sf,Q2))
    out = Piterm + Pfterm
    return out

def Kmatrixinv(s,e):
    top = 4*pow(g,2)*pow(mr,2)
    bottom = 3*xi*(pow(mr,2)-s)
    return bottom/top

def A_mu(P,Q2):
    s = pf.dotprod(P,P)
    if s > 4*m1**2:
        return ((3*P)/(2*g**2*mr**2)) - np.real(Gvec(P,P,0))
    if s < 4*m1**2:
        a = ((3*P)/(2*g**2*mr**2)) + (P*(4*m1**2)/(32*math.pi*pow(s,3/2)*np.abs(qStar(s)))) - np.real(Gvec(P,P,0))
        if np.imag(a[0]) < .000001:
            np.imag(a)==np.array([0,0,0,0])
        return a

def A_muII(P,Q2):
    s = pf.dotprod(P,P)
    if s > 4*m1**2:
        return ((3*P)/(2*g**2*mr**2)) - np.real(GvecII(P,P,0))
    if s < 4*m1**2:
        a = ((3*P)/(2*g**2*mr**2)) + (P*(4*m1**2)/(32*math.pi*pow(s,3/2)*np.abs(qStar(s)))) - np.real(GvecII(P,P,0))
        if np.imag(a[0]) < .000001:
            np.imag(a)==np.array([0,0,0,0])
        return a


def Aparam(Pi,Pf,Q2):
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)
    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)
    lead = mr**2/(mr**2+Q2)
    t1 = (.5+(Bcon*(si-sf)))
    t2 = (.5+(Bcon*(sf-si)))
    Fsf = A_mu(Pf,Q2)
    Fsi = A_mu(Pi,Q2)
    return lead*(t1*Fsi+t2*Fsf)

def AparamII(Pi,Pf,Q2):
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)
    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)
    lead = mr**2/(mr**2+Q2)
    t1 = (.5+(Bcon*(si-sf)))
    t2 = (.5+(Bcon*(sf-si)))
    Fsf = A_muII(Pf,Q2)
    Fsi = A_muII(Pi,Q2)
    return lead*(t1*Fsi+t2*Fsf)

def A_term_I(Pi,Pf,Q2):
    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)
    return MI(si)*Aparam(Pi,Pf,Q2)*MI(sf)

def A_term_II(Pi,Pf,Q2):
    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)
    return MII(si)*AparamII(Pi,Pf,Q2)*MII(sf)

def f(Q2):
    bot = mr**2 + Q2
    return mr**2 / bot
#
def GI_term(Pi,Pf,Q2):
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)
    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)
    return MI(si)*f(Q2)*Gvec(Pi,Pf,Q2)*MI(sf)

def GII_term(Pi,Pf,Q2):
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)
    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)
    return MII(si)*f(Q2)*GvecII(Pi,Pf,Q2)*MII(sf)

def WDF_I_imag(Pi,Pf,Q2):
    t = np.imag(GI_term(Pi,Pf,Q2) + A_term_I(Pi,Pf,Q2))
    return t

def WDF_I_real(Pi,Pf,Q2):
    t = np.real(GI_term(Pi,Pf,Q2) + A_term_I(Pi,Pf,Q2))
    return t

def WDF_II_imag(Pi,Pf,Q2):
    t = np.imag(GII_term(Pi,Pf,Q2) + A_term_II(Pi,Pf,Q2))
    return t

def WDF_II_real(Pi,Pf,Q2):
    t = np.real(GII_term(Pi,Pf,Q2) + A_term_II(Pi,Pf,Q2))
    return t

data1swdfi = []
data1swdfr = []
data2swdfi = []
data2swdfr = []
srange = []
srangee = []

# print(efrange)
RElist = [1.5,2.5]
# print(RElist)

gval = [.1,.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
for gv in gval:
    print(gv)
    a = 1
    # print("a = " + str(1))
    # print("a = " + str(a))
    b = -(2*mr**2)
    # print("b = " + str(-2*mr**2))
    # print("b = " + str(b))
    c = (mr**4 + (mr**4*gv**4)/(144*(math.pi)**2))
    # print("c = " + str(mr**4 + (mr**4*gv**4)/(144*(math.pi)**2)))
    # print("c = " + str(c))
    d = -(mr**4*gv**4/(36*(math.pi)**2))
    # print("d = " + str(-(mr**4*gv**4/(36*(math.pi)**2))))
    # print("d = " + str(d))
    coeff = [a,b,c,d]
    roots = np.roots(coeff)
    print(roots)


# for i in RElist:
#     testrange2 = np.linspace(i+disp*1j,10,1)
#     # print(testrange2)
#     for ef in testrange2:
#         P1 = np.array([ei,0,0,0])
#         P2 = np.array([ef,0,0,0])
#
#         b2 = np.subtract(P2,P1)
#         sd2 = pf.dotprod(b2,b2)
#         Q2p = -1*(sd2)
#
#         print(Q2p)
#         print(pf.dotprod(P1,P1))
#         print(pf.dotprod(P2,P2))
#         print("si: " +str(MI(pf.dotprod(P1,P1))))
#         print("sf: " +str(MI(pf.dotprod(P2,P2))))
#         print("A real: " +str(np.real(Aparam(P1,P2,Q2p)[0])))
#         print("A imag: " +str(np.imag(Aparam(P1,P2,Q2p)[0])))
#         print("G real: " +str(np.real(Gvec(P1,P2,Q2p)[0])))
#         print("G imag: " +str(np.imag(Gvec(P1,P2,Q2p)[0])))
#
#
#         print("Real Component of Wdf in the first sheet: " + str(WDF_I_real(P1,P2,Q2p)[0]))
#         print("Imag Component of Wdf in the first sheet: " + str(WDF_I_imag(P1,P2,Q2p)[0]))
#         print("Real Component of Wdf in the second sheet: " + str(WDF_II_real(P1,P2,Q2p)[0]))
#         print("Imag Component of Wdf in the second sheet: " + str(WDF_II_imag(P1,P2,Q2p)[0]))
#         print("Energy value: " + str(testrange2))
