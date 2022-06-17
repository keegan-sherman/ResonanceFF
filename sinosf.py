# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import pylab
import numpy as np
import mpmath
import physfunc as pf
import math
import cmath
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
# disp = 0.47915701951257783
# e = 0

ei = 1.05*(m1+m2)
# si = 0
# Q2 = 1.5
eps = 0
# sfrange = np.linspace(0+disp*1j,9+disp*1j,1000)
efrange = np.linspace(1+disp*1j,3+disp*1j,1000)
# srange2 = np.linspace(1+disp1*1j,9+disp1*1j,1000)
# Pi = np.array([ei,0,0,0])
# si = pf.dotprod(Pi,Pi)
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
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)
    return (IaFr(si,sf,Q2) + 1j*IaFi(si,sf,Q2))

def GII(si,sf,Q2):
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)
    return (IaFr(si,sf,Q2) - 1j*IaFi(si,sf,Q2))

#combination of vector GS
def ComboG_a(si,sf,Q2):
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)
    return (IaFr1(si,sf,Q2) + 1j*IaFi1(si,sf,Q2))

def ComboG_b(si,sf,Q2):
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)
    return (IaFr2(si,sf,Q2) + 1j*IaFi2(si,sf,Q2))

def ComboG_aII(si,sf,Q2):
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)
    return (IaFr1(si,sf,Q2) - 1j*IaFi1(si,sf,Q2))

def ComboG_bII(si,sf,Q2):
    b = np.subtract(Pf,Pi)
    s = pf.dotprod(b,b)
    Q2=-1*(s)
    return (IaFr2(si,sf,Q2) - 1j*IaFi2(si,sf,Q2))

# def GIvec2ndterm(sf,Q2):
#     b = np.subtract(Pf,Pi)
#     s = pf.dotprod(b,b)
#     Q2=-1*(s)
#     sf = pf.dotprod(Pf,Pf)
#     return Pf*ComboG_a(sf,Q2) + Pi*ComboG_b(sf,Q2)
#
# def GvecI(sf):
#     b = np.subtract(Pf,Pi)
#     s = pf.dotprod(b,b)
#     Q2=-1*(s)
#     return (Pi+Pf)*GI(sf,Q2) - 2*GIvec2ndterm(sf,Q2)
#     # return (2*np.sqrt(sf))*GI(sf) - 2*GIvec2ndterm(sf)

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
        a = ((3*P)/(2*g**2*mr**2)) + (1j*P*(4*m1**2)/(32*math.pi*pow(s,3/2)*(qStar(s)))) - np.real(Gvec(P,P,0))
        if np.imag(a[0]) < .000001:
            np.imag(a)==np.array([0,0,0,0])
        return a

    # return -(1j*P*(4*m1**2)/(32*math.pi*pow(s,3/2)*(qStar(s))))

def A_muII(P,Q2):
    s = pf.dotprod(P,P)
    if s > 4*m1**2:
        return ((3*P)/(2*g**2*mr**2)) - np.real(GvecII(P,P,0))
    if s < 4*m1**2:
        a = ((3*P)/(2*g**2*mr**2)) + (1j*P*(4*m1**2)/(32*math.pi*pow(s,3/2)*(qStar(s)))) - np.real(GvecII(P,P,0))
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
    # Fsf = 2*Q0*Pf*pf.dbydx(Kmatrixinv,sf,ds,(e)) - Q0*np.real(Gvec(Pf,Pf,0))
    # Fsi = 2*Q0*Pi*pf.dbydx(Kmatrixinv,si,ds,(e)) - Q0*np.real(Gvec(Pi,Pi,0))
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
    # Fsf = 2*Q0*Pf*pf.dbydx(Kmatrixinv,sf,ds,(e)) - Q0*np.real(Gvec(Pf,Pf,0))
    # Fsi = 2*Q0*Pi*pf.dbydx(Kmatrixinv,si,ds,(e)) - Q0*np.real(Gvec(Pi,Pi,0))
    return lead*(t1*Fsi+t2*Fsf)

# def GvecII(sf):
#     si = sf
#     Pi = np.array([np.sqrt(si),0,0,0])
#     Pf = np.array([np.sqrt(sf),0,0,0])
#     step1 = (Pi[0]+Pf[0])*GI(sf) - 2*GIvec2ndterm(sf)
#     return step1 - 2j*np.imag(step1)

# def GII(sf):
#     return (GI(sf) - 2j*np.imag(GI(sf)))

def A_term_I(Pi,Pf,Q2):
    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)
    return MI(si)*Aparam(Pi,Pf,Q2)*MI(sf)

def A_term_II(Pi,Pf,Q2):
    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)
    return MII(si)*AparamII(Pi,Pf,Q2)*MII(sf)
#
# def A_term_II(sf):
#     return (MII(sf))*ascale*(MII(sf))

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
#
# def GII_term(sf):
#     return (MII(sf))*GII(sf)*(MII(sf))
#
def WDF_I_imag(Pi,Pf,Q2):
    t = np.imag(GI_term(Pi,Pf,Q2) + A_term_I(Pi,Pf,Q2))
    return t

def WDF_I_real(Pi,Pf,Q2):
    t = np.real(GI_term(Pi,Pf,Q2) + A_term_I(Pi,Pf,Q2))
    return t
#
# def WDF_II_imag(sf):
#     t = np.imag(GII_term(sf) + A_term_II(sf))
#     return t
#
# def WDF_II_real(sf):
#     t = np.real(GII_term(sf) + A_term_II(sf))
#     return t

def WDF_II_imag(Pi,Pf,Q2):
    t = np.imag(GII_term(Pi,Pf,Q2) + A_term_II(Pi,Pf,Q2))
    return t

def WDF_II_real(Pi,Pf,Q2):
    t = np.real(GII_term(Pi,Pf,Q2) + A_term_II(Pi,Pf,Q2))
    return t

# def Kinvder(s):
#     return (6*math.pi)/(g**2*mr**2*np.sqrt(s))

data1swdfi = []
data1swdfr = []
data2swdfi = []
data2swdfr = []
srange = []
srangee = []

P1 = np.array([pow(ei,2),0,0,0])
P2 = np.array([2.14,0,0,0])



# print(IaFr(si,4.2,1.3))
# print(IaFi(si,4.2,1.3))
# print(IaFr1(si,4.2,1.3))
# print(IaFi1(si,4.2,1.3))
# print(IaFr2(si,4.2,1.3))
# print(IaFi2(si,4.2,1.3))
# print(realF(.5,si,4.2,1.3))
# print(imagF(.5,si,4.2,1.3))
# print(Lminus(.5,m1,m2,1.3,si,4.2,eps))
# print(Lplus(.5,m1,m2,1.3,si,4.2,eps))
# print(yminus(.5,m1,m2,1.3,si,4.2,eps))
# print(yplus(.5,m1,m2,1.3,si,4.2,eps))
# print(MI(4.8-.25j))
# print(MI_imag(4.0-.25j))
# print(MItestfunc(4.0-.25j))
# print(GI_term(4.0-.25j))
# print(A_term_I(4.0-.25j))
# print(WDF_II_real(4.1))
# print(WDF_II_imag(4.1))
# P1T = np.array([4.4,0,0,0])
# P2T = np.array
# print(WDF_I_real(P1T,P2,Q2p)[0])
# print(WDF_I_imag(P1T,P2,Q2p)[0])


# b11 = np.subtract(np.array([2.5,0,0,0]),np.array([ei,0,0,0]))
# sd1 = pf.dotprod(b11,b11)
# Q21=-1*(sd1)
# print(WDF_I_real(np.array([ei,0,0,0]),np.array([2.5,0,0,0]),Q21))
# P1T = np.array([ei,0,0,0])
# P2T = np.array([1.1+.11*1j,0,0,0])
# b11 = np.subtract(P2,P1)
# sd1 = pf.dotprod(b11,b11)
# Q21=-1*(sd1)
# print(WDF_I_real(P1T,P2T,Q21))
# print(WDF_I_imag(P1T,P2T,Q21))

testrange = np.linspace(1.1+.479j,1.11+.479j,1)

# print(efrange)

for ef in efrange:
    # P = np.array([np.sqrt(pow(ef,2)+pow(Pz,2)),0,0,Pz])
    # Pf = np.array([np.sqrt(pow(ef,2)),0,0,0])
    # Pi = np.array([np.sqrt(pow(ei,2)),0,0,0])
    P = np.array([np.sqrt(pow(ef,2)),0,0,0])
    Pf = P
    Pi = P
    P_i = np.array([ei,0,0,0])
    P1 = np.array([ei,0,0,0])
    P2 = np.array([1.1+.479j,0,0,0])
    # sf = pf.dotprod(Pf,Pf)
    # sfrange.append(sf)
    s_1 = pf.dotprod(P,P)
    # s_2 = s_1 +disp1*1j
    srange.append(s_1)
    # for i in srange:
    #     srangee.append(srange[i]+disp*1j)

    b = np.subtract(Pf,Pi)
    b2 = np.subtract(P2,P1)
    sd = pf.dotprod(b,b)
    sd2 = pf.dotprod(b2,b2)
    Q2=-1*(sd)
    Q2p = -1*(sd2)
    # G = Gvec(P_i,P_i,0)[0]
    # A_G = A_mu(P,0)[0]
    # Ap = Aparam(P_i,P,Q2)[0]
    # Amu = AparamII(P_i,P,Q2)
    # Wdfr = WDF_I_real(P_i,P,Q2)[0]
    # Wdfi = WDF_I_imag(P_i,P,Q2)[0]

    Wdfr2 = WDF_II_real(P_i,P,Q2)[0]
    Wdfi2 = WDF_II_imag(P_i,P,Q2)[0]

    s2f = pf.dotprod(P2,P2)
    s2I = pf.dotprod(P1,P1)
    # print(MI(s2I))
    # print(MI(s2f))
    # print(Aparam(P1,P2,Q2p))
    # print(Gvec(P1,P2,Q2p))
    # print(WDF_I_real(P1,P2,Q2p))
    # print(WDF_I_imag(P1,P2,Q2p))
    # print(f(Q2p))

    #
    # data1swdfi.append(WDF_I_imag(sf))
    # data1swdfr.append(WDF_I_real(sf))
    # # data2swdfi.append(WDF_II_imag(sf))
    # # data2swdfr.append(WDF_II_real(sf))
    # # data2swdfr.append(np.real(ComboG_b(si,si,Q2)))
    # # data2swdfi.append(np.imag(ComboG_b(si,si,Q2)))
    # data2swdfr.append(np.real(G))
    # data2swdfi.append(np.imag(G))
    # data2swdfr.append(np.real(Ap))
    # data2swdfi.append(np.imag(Ap))

    data2swdfr.append(np.real(Wdfr2))
    data2swdfi.append(np.real(Wdfi2))

    # data2swdfr.append(np.real(GvecII(Pi,Pf,Q2))[0])
    # data2swdfi.append(np.imag(GvecII(Pi,Pf,Q2))[0])
    # data2swdfr.append(np.real((MI(sf)*(Gvec(Pi,Pf,Q2))[3])*MI(si))+(MI(sf)*Aparam(Pi,Pf,Q2)[3]*MI(si)))
    # data2swdfi.append(np.imag((MI(sf)*(Gvec(Pi,Pf,Q2))[3])*MI(si))+(MI(sf)*Aparam(Pi,Pf,Q2)[3]*MI(si)))
    # data2swdfr.append(np.real(Kinvder(sf)))
    # data2swdfi.append(np.imag(Kinvder(sf)))
    # data2swdfr.append(np.real(A_G))
    # # data2swdfi.append(np.imag(A_G))
    # data2swdfi.append(0)

# print(data)
# plt.plot(sfrange,data4)
# plt.show()
# print(srange)


# print("G1 =" +str(ComboG_a(4.4,4.4,0)))


mpl.rcParams['mathtext.rm'] = 'Yrsa'
mpl.rcParams['mathtext.it'] = 'Yrsa:italic'
mpl.rcParams['mathtext.bf'] = 'Yrsa:bold'

fig, plots2 = plt.subplots(2,1)
plots2[0].axhline(0, color = 'k')
plots2[1].axhline(0, color = 'k')
# plots2[0]
# plots2[0].axvline(4, color = 'k')
# plots2[1].axvline(4, color = 'k')
# plots2[0].set_ylim([-15000,15000])
# plots2[1].set_ylim([-15000,15000])
# plots2[0].plot(sfrange,data2swdfr, color = 'darkorange')
# plots2[1].plot(sfrange,data2swdfi, color = 'darkorange')
plots2[0].plot(efrange,data2swdfr, color = 'darkgreen')
plots2[1].plot(efrange,data2swdfi, color = 'darkgreen')
# plots2[0].set_title(r'$\mathcal{A}^0$ along the real axis in the first sheet, $P_i \neq P_f$')
plots2[0].set_title(r'$\mathcal{W}_{df}^0$ on the real axis in the second sheet, $P_i \neq P_f$')
# plots2[0].set_title(r'$\mathcal{A}$ along the real axis in the first sheet, $P_{i} = P_{f} = P = [E,0,0,0]$')
# plots2[0].set_title(r'$\mathcal{G}$ along the real axis in the first sheet, $P_{i} = P_{f} = P = [(E^2+P_z^2)^{1/2},0,0,Pz]$, $P_z = 2\pi /15$',fontsize=9)
# plots2[0].text(.9,.9,"Im[sf]="+str(disp),transform=plots2[0].transAxes,size=7)
# plots2[0].set('Oth component of G, 1st/ Sheet')
plots2[1].set(xlabel='$e_f$')
plots2[0].set_ylabel(ylabel='$Re[\mathcal{W}_{df}^0]$',fontname="Yrsa",size=10)
plots2[1].set_ylabel(ylabel='$Im[\mathcal{W}_{df}^0]$',fontname="Yrsa",size=10)
# plots2[0].set_ylabel(ylabel='$Re[\mathcal{A}^0]$',fontname="Yrsa",size=10)
# plots2[1].set_ylabel(ylabel='$Im[\mathcal{A}^0]$',fontname="Yrsa",size=10)
#
plt.show()
