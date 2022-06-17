from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pylab
import numpy as np
import mpmath
import physfunc as pf
import math
import cmath
import scipy
from scipy.integrate import quad
import matplotlib.colors
import matplotlib as mpl
import sys
import time
start_time = time.time()

# print(sys.getrecursionlimit())

m1 = 1
m2 = 1
g = 3
mr = 2.2
xi = 1/2
ascale = .0001
Bcon = .1

ei = 1.05*(m1+m2)
# Pi = np.array([ei,0,0,0])
# si = pow(ei,2)
si = 4.4
# si = 0
# Q2 = 0
eps = 0

def csqrt(z):
    arg = np.angle(z)
    norm = np.absolute(z)
    newarg = np.mod(arg + 2*math.pi, 2*math.pi)
    return np.sqrt(norm)*np.exp(1j*newarg/2)

def fourroot(z):
    arg = np.angle(z)
    norm = np.absolute(z)
    newarg = np.mod(arg + 2*math.pi, 2*math.pi)
    return pow(norm,1/4)*np.exp(1j*newarg/4)

def csqrt2nd(z):
    arg = np.angle(z)
    norm = np.absolute(z)
    newarg = np.mod(arg + 2*math.pi, 2*math.pi) + 2*math.pi
    return np.sqrt(norm)*np.exp(1j*newarg/2)

def frt2(z):
    arg = np.angle(z)
    norm = np.absolute(z)
    newarg = np.mod(arg + 2*math.pi, 2*math.pi) +2*math.pi
    return pow(norm,1/4)*np.exp(1j*newarg/4)

def frt3(z):
    arg = np.angle(z)
    norm = np.absolute(z)
    newarg = np.mod(arg + 2*math.pi, 2*math.pi) +4*math.pi
    return pow(norm,1/4)*np.exp(1j*newarg/4)

def frt4(z):
    arg = np.angle(z)
    norm = np.absolute(z)
    newarg = np.mod(arg + 2*math.pi, 2*math.pi) +6*math.pi
    return pow(norm,1/4)*np.exp(1j*newarg/4)

def testfunction(z):
    return 2/(z**2+3) - 4

# def csqrtcombo(z):
#     return csqrt(z) + csqrt2nd(z)

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

def qStar2(s,m1,m2):
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

#EFFORTS FOR G
#si != sf imaginary
#then si = sf later

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

def realF(x,sf):
    si = sf
    #only true for no 3 momentum
    b = np.sqrt(sf)-np.sqrt(si)
    sa = pow(b,2)
    Q2=-1*(sa)
    sth = pow((m1+m2),2)
    fpi = 4*cmath.pi
    fpisq = pow(fpi,2)
    lead = 1 / (fpisq*si)
    numf = (Lplus(x,m1,m2,Q2,si,sf,eps)-Lminus(x,m1,m2,Q2,si,sf,eps))
    denf = (yplus(x,m1,m2,Q2,si,sf,eps)-yminus(x,m1,m2,Q2,si,sf,eps))
    fracf = numf/denf
    return np.real(lead * fracf)

def imagF(x,sf):
    si = sf
    b = np.sqrt(sf)-np.sqrt(si)
    sa = pow(b,2)
    Q2=-1*(sa)
    sth = pow((m1+m2),2)
    fpi = 4*cmath.pi
    fpisq = pow(fpi,2)
    lead = 1 / (fpisq*si)
    numf = (Lplus(x,m1,m2,Q2,si,sf,eps)-Lminus(x,m1,m2,Q2,si,sf,eps))
    denf = (yplus(x,m1,m2,Q2,si,sf,eps)-yminus(x,m1,m2,Q2,si,sf,eps))
    fracf = numf/denf
    return np.imag(lead * fracf)

##############################################################################
#from sinosf

def realFstar(x,si,sf,Q2):
    sth = pow((m1+m2),2)
    fpi = 4*cmath.pi
    fpisq = pow(fpi,2)
    lead = 1 / (fpisq*si)
    numf = (Lplus(x,m1,m2,Q2,si,sf,eps)-Lminus(x,m1,m2,Q2,si,sf,eps))
    denf = (yplus(x,m1,m2,Q2,si,sf,eps)-yminus(x,m1,m2,Q2,si,sf,eps))
    fracf = numf/denf
    return np.real(lead * fracf)

def imagFstar(x,si,sf,Q2):
    sth = pow((m1+m2),2)
    fpi = 4*cmath.pi
    fpisq = pow(fpi,2)
    lead = 1 / (fpisq*si)
    numf = (Lplus(x,m1,m2,Q2,si,sf,eps)-Lminus(x,m1,m2,Q2,si,sf,eps))
    denf = (yplus(x,m1,m2,Q2,si,sf,eps)-yminus(x,m1,m2,Q2,si,sf,eps))
    fracf = numf/denf
    return np.imag(lead * fracf)

def realF11(x,si,sf,Q2):
    return x * realFstar(x,si,sf,Q2)

def imagF11(x,si,sf,Q2):
    return x * imagFstar(x,si,sf,Q2)

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

################################################################################################
#####################
#this is the TRUE M
def MI(sf):
    return MI_real(sf) +(1j*(MI_imag(sf)))

def MII(sf):
    return MII_real(sf) + (1j*(MII_imag(sf)))

def IaFr(sf):
    IaR,error = scipy.integrate.quad(realF,0,1,args=(sf))
    return (IaR)

def IaFi(sf):
    IaI,error2 = scipy.integrate.quad(imagF,0,1,args=(sf))
    return (IaI)


#############################################
def IaFrstar(si,sf,Q2):
    IaR,error = scipy.integrate.quad(realFstar,0,1,args=(si,sf,Q2))
    return (IaR)

def IaFistar(si,sf,Q2):
    IaI,error2 = scipy.integrate.quad(imagFstar,0,1,args=(si,sf,Q2))
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

#############################################################################################

def GI(sf):
    return (IaFr(sf) + 1j*IaFi(sf))

def GII(sf):
    return (IaFr(sf) - 1j*IaFi(sf))

def GIforplot(s):
    t = np.real(GI(s))
    if t > .03:
        return .03
    if t < -.03:
        return -.03
    if t == -math.inf:
        return -.03
    elif t == math.inf:
        return .03
    return t

def GI2forplot(s):
    t = np.imag(GI(s))
    if t > .03:
        return .03
    if t < -.03:
        return -.03
    if t == -math.inf:
        return -.03
    elif t == math.inf:
        return .03
    return t

def GIIforplot(s):
    t = np.real(GII(s))
    if t > .03:
        return .03
    if t < -.03:
        return -.03
    if t == -math.inf:
        return -.03
    elif t == math.inf:
        return .03
    return t

def GII2forplot(s):
    t = np.imag(GII(s))
    if t > .03:
        return .03
    if t < -.03:
        return -.03
    if t == -math.inf:
        return -.03
    elif t == math.inf:
        return .03
    return t

def A_term_I(sf):
    si = sf
    return MI(sf)*ascale*MI(si)

def A_term_II(sf):
    si =sf
    return MII(sf)*ascale*MII(si)

def GI_term(sf):
    si=sf
    return MI(sf)*GI(sf)*MI(si)

def GII_term(sf):
    si=sf
    return MII(sf)*GII(sf)*MII(si)

def WDF_I_imag(sf):
    t = np.imag(GI_term(sf) + A_term_I(sf))
    if t > 700:
        return 700
    if t < -700:
        return -700
    if t == -math.inf:
        return -700
    elif t == math.inf:
        return 700
    return t

def WDF_I_real(sf):
    t = np.real(GI_term(sf) + A_term_I(sf))
    if t > 700:
        return 700
    if t < -700:
        return -700
    if t == -math.inf:
        return -700
    elif t == math.inf:
        return 700
    return t

def WDF_II_imag(sf):
    t = np.imag(GII_term(sf) + A_term_II(sf))
    if t > 7000:
        return 7000
    if t < -7000:
        return -7000
    if t == -math.inf:
        return -7000
    elif t == math.inf:
        return 7000
    return t

def WDF_II_real(sf):
    t = np.real(GII_term(sf) + A_term_II(sf))
    if t > 7000:
        return 7000
    if t < -7000:
        return -7000
    if t == -math.inf:
        return -7000
    elif t == math.inf:
        return 7000
    return t

###################################################################
#Here is the vector G information from sinosf converted to 1d variables

def vecGI(si,sf,Q2):
    return (IaFrstar(si,sf,Q2) + 1j*IaFistar(si,sf,Q2))

def vecGII(si,sf,Q2):
    return (IaFrstar(si,sf,Q2) - 1j*IaFistar(si,sf,Q2))

#combination of vector GS
def vecComboG_a(si,sf,Q2):
    return (IaFr1(si,sf,Q2) + 1j*IaFi1(si,sf,Q2))

def vecComboG_b(si,sf,Q2):
    return (IaFr2(si,sf,Q2) + 1j*IaFi2(si,sf,Q2))

def vecComboG_aII(si,sf,Q2):
    return (IaFr1(si,sf,Q2) - 1j*IaFi1(si,sf,Q2))

def vecComboG_bII(si,sf,Q2):
    return (IaFr2(si,sf,Q2) - 1j*IaFi2(si,sf,Q2))

def vecGvec(si,sf,Q2):
    Piterm = np.sqrt(si)*(vecGI(si,sf,Q2)-2*vecComboG_b(si,sf,Q2))
    Pfterm = np.sqrt(sf)*(vecGI(si,sf,Q2)-2*vecComboG_a(si,sf,Q2))
    out = Piterm + Pfterm
    return out

def vecGvecII(si,sf,Q2):
    Piterm = np.sqrt(si)*(vecGII(si,sf,Q2)-2*vecComboG_bII(si,sf,Q2))
    Pfterm = np.sqrt(sf)*(vecGII(si,sf,Q2)-2*vecComboG_aII(si,sf,Q2))
    out = Piterm + Pfterm
    return out

def vecA_mu(s,Q2):
    P = np.sqrt(s)
    if np.abs(s) > 4*m1**2:
        return ((3*P)/(2*g**2*mr**2)) - np.real(vecGvec(s,s,0))
    if np.abs(s) < 4*m1**2:
        a = ((3*P)/(2*g**2*mr**2)) + (1j*P*(4*m1**2)/(32*math.pi*pow(s,3/2)*(qStar(s)))) - np.real(vecGvec(s,s,0))
        return a

def vecA_muII(s,Q2):
    P = np.sqrt(s)
    if np.abs(s) > 4*m1**2:
        return ((3*P)/(2*g**2*mr**2)) - np.real(vecGvecII(s,s,0))
    if np.abs(s) < 4*m1**2:
        a = ((3*P)/(2*g**2*mr**2)) + (1j*P*(4*m1**2)/(32*math.pi*pow(s,3/2)*(qStar(s)))) - np.real(vecGvecII(s,s,0))
        return a


def vecAparam(si,sf,Q2):
    lead = mr**2/(mr**2+Q2)
    t1 = (.5+(Bcon*(si-sf)))
    t2 = (.5+(Bcon*(sf-si)))
    Fsf = vecA_mu(sf,Q2)
    Fsi = vecA_mu(si,Q2)
    return lead*(t1*Fsi+t2*Fsf)

def vecAparamII(si,sf,Q2):
    lead = mr**2/(mr**2+Q2)
    t1 = (.5+(Bcon*(si-sf)))
    t2 = (.5+(Bcon*(sf-si)))
    Fsf = vecA_muII(sf,Q2)
    Fsi = vecA_muII(si,Q2)
    return lead*(t1*Fsi+t2*Fsf)

def vecA_term_I(si,sf,Q2):
    return MI(si)*vecAparam(si,sf,Q2)*MI(sf)

def vecA_term_II(si,sf,Q2):
    return MII(si)*vecAparamII(si,sf,Q2)*MII(sf)

#this is f bu to ensure it uses this function its called f11
def f11(Q2):
    bot = mr**2 + Q2
    return mr**2 / bot

def vecGI_term(si,sf,Q2):
    return MI(si)*f11(Q2)*vecGvec(si,sf,Q2)*MI(sf)

def vecGII_term(si,sf,Q2):
    # si = 4.4
    # pf_hold = np.sqrt(sf)
    # pi_hold = np.sqrt(si)
    # b = pf_hold - pi_hold
    # s2 = b**2
    # Q2=-1*(s2)
    return MII(si)*f11(Q2)*vecGvecII(si,sf,Q2)*MII(sf)

def vecWDF_I_imag(sf):
    si = 4.4
    pf_hold = np.sqrt(sf)
    pi_hold = np.sqrt(si)
    b = pf_hold - pi_hold
    s2 = b**2
    Q2=-1*(s2)
    t = np.imag(vecGI_term(si,sf,Q2) + vecA_term_I(si,sf,Q2))
    return t

def vecWDF_I_real(sf):
    si = 4.4
    pf_hold = np.sqrt(sf)
    pi_hold = np.sqrt(si)
    b = pf_hold - pi_hold
    s2 = b**2
    Q2=-1*(s2)
    t = np.real(vecGI_term(si,sf,Q2) + vecA_term_I(si,sf,Q2))
    return t

def vecWDF_II_imag(sf):
    si = 4.4
    pf_hold = np.sqrt(sf)
    pi_hold = np.sqrt(si)
    b = pf_hold - pi_hold
    s2 = b**2
    Q2=-1*(s2)
    t = np.imag(vecGII_term(si,sf,Q2) + vecA_term_II(si,sf,Q2))
    return t

def vecWDF_II_real(sf):
    si = 4.4
    pf_hold = np.sqrt(sf)
    pi_hold = np.sqrt(si)
    b = pf_hold - pi_hold
    s2 = b**2
    Q2=-1*(s2)
    t = np.real(vecGII_term(si,sf,Q2) + vecA_term_II(si,sf,Q2))
    return t

def vecWdfI(z):
    si = 4.4
    pf_hold = np.sqrt(z)
    pi_hold = np.sqrt(si)
    b = pf_hold - pi_hold
    s2 = b**2
    Q2=-1*(s2)
    return vecGI_term(si,z,Q2) + vecA_term_I(si,z,Q2)

def vecWdfII(z):
    si = 4.4
    pf_hold = np.sqrt(z)
    pi_hold = np.sqrt(si)
    b = pf_hold - pi_hold
    s2 = b**2
    Q2=-1*(s2)
    return vecGII_term(si,z,Q2) + vecA_term_II(si,z,Q2)

# poles = []
# poles2 = []
# a = []
#
# k = np.linspace(.4,.5,100)
# ja = np.linspace(4,5,100)

# for i in k:
#     for m in ja:
#         a.append(m+1j*i)
# b = np.linspace(.05,1,1000)
# k = np.linspace(0,1,1000)

# for i in a:
#     wdf = vecWdfII(k)
#     if wdf == np.inf:
#         poles.append(i)

# psd = [(4.717171717171717+0.47575757575757577j), (4.717171717171717+0.4767676767676768j), (4.717171717171717+0.4777777777777778j), (4.717171717171717+0.47878787878787876j), (4.717171717171717+0.4797979797979798j), (4.717171717171717+0.4808080808080808j), (4.717171717171717+0.4818181818181818j), (4.717171717171717+0.48282828282828283j)]


# print(a)
# print(len(a))

# for h in psd:
#     wdf = vecWdfII(h)
#     if np.abs(wdf) >= 470000:
#         poles2.append(h)
#     # if wdf == -np.inf:
#     #     poles.append(h)
#
# print(poles2)


# for i in psd:
#     if
# print(vecWdfII(4.67 + .47915701951257783*1j))
# print(vecWdfII(4.68 + .47915701951257783*1j))
# print(vecWdfII(4.69 + .47915701951257783*1j))
# print(vecWdfII(4.70 + .47915701951257783*1j))
# print(vecWdfII(4.71 + .47915701951257783*1j))
# print(vecWdfII(4.72 + .47915701951257783*1j))
# print(vecWdfII(4.73 + .47915701951257783*1j))
# print(vecWdfII(4.74 + .47915701951257783*1j))
# print(vecWdfII(4.75 + .47915701951257783*1j))

""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""

# def integrate_on_contour(function,path):
#     result = 0.0
#     for n in np.arange(len(path)-1):
#         z0 = path[n]
#         dz = path[n+1]-path[n]
#         real_integrand = lambda x: np.real(function(z0+x*dz)*dz)
#         imag_integrand = lambda x: np.imag(function(z0+x*dz)*dz)
#         real_result = scipy.integrate.quad(real_integrand,0.0,1.0)[0]
#         imag_result = scipy.integrate.quad(imag_integrand,0.0,1.0)[0]
#         result += real_result + 1j*imag_result
#     return result

# def contourtest(z):
#     return 3/(z-(4.05+.25j))

# C1 = 4.721 - .48j
# C2 = 4.722 - .48j
# C3 = 4.722 - .479j
# C4 = 4.721 - .479j
# C5 = C1

# B1 = 4.721 + .479j
# B2 = 4.722 + .479j
# B3 = 4.722 + .48j
# B4 = 4.721 + .48j
# B5 = B1

# D1 = .2365 - .001*1j
# D2 = .2365 + .001*1j
# D3 = .2375 + .001*1j
# D4 = .2375 - .001*1j
# D5 = D1

# C = [C1,C2,C3,C4,C5]
# BA = [B1,B2,B3,B4,B5]
# D = [D5,D4,D3,D2,D1]


# # contour integration
# integral1 = integrate_on_contour(lambda z: MII(z),C)
# integral = integrate_on_contour(lambda x: MII(x),BA)
# integral2 = integrate_on_contour(lambda z: MII(z),D)

# print(r'Res(4.72 + .479i) = ' + str(integral/(2*math.pi*1j)))
# print(r'Res(4.72 - .479i) = ' + str(integral1/(2*math.pi*1j)))
# print(r'Res(.237) = ' + str(integral2/(2*math.pi*1j)))

# f = lambda z: np.real(csqrt(z))
# g = lambda z: np.real(csqrt2nd(z))
# f = lambda z: np.real(np.log(z))
# g = lambda z: np.real(np.log(z)+2*math.pi*1j)
# f = lambda z: IaF(z)
# f = lambda z: arctangent(z)
# f = lambda z: MI_real(z)
# f = lambda z: MI_imag(z)
f = lambda z: MII_real(z)
# f = lambda z: MII_imag(z)
# f = lambda z: Gsingplot(z)
# f = lambda z: WDF_I_imag(z)
# f = lambda z: WDF_II_imag(z)
# f = lambda z: WDF_I_real(z)
# f = lambda z: WDF_II_real(z)
# f = lambda z: IaF(z)
# f = lambda z: np.real(GII(z))
# f = lambda z: GI_term_r(z)
# f = lambda z: (GIforplot(z))
# f = lambda z: (GI2forplot(z))
# f = lambda z: (GIIforplot(z))
# f = lambda z: (GII2forplot(z))
# f = lambda z: (vecWDF_II_imag(z))
# f = lambda z: vecWDF_I_imag(z)
# f = lambda z: np.real(testfunction(z))
# f = lambda z: np.imag(testfunction(z))
# f = lambda z: fourroot(z)
# g = lambda z: frt2(z)
# b = lambda z: frt3(z)
# t = lambda z: frt4(z)

cmap = mpl.colors.LinearSegmentedColormap.from_list('laneys cmap',colors=['indigo','darkviolet','blueviolet','mediumslateblue','royalblue','cornflowerblue','lightseagreen', 'mediumseagreen', 'lightgreen', 'greenyellow','yellow','gold','orange','darkorange','orangered','red','firebrick'],N=200)

fig = pylab.figure()
ax = Axes3D(fig)
X = np.arange(-5,5,0.125)
# X = np.arange(4.025, 10.0, 0.025)
# X = np.arange(0.025,3.975,0.025)
Y = np.arange(-5, 5, 0.125)


# Xlow = np.arange(-1,.025,0.025)
# Xhigh = np.arange(-0.025,1,.025)
# Ylow = np.arange(-1,0,0.025)
# KL = np.arange(0, 1, 0.0125)
# Yhigh = []
# for i in KL:
#     Yhigh = KL
#
# Yhigh[0]=np.nan
# # print(Yhigh[0])
#
# X1,Y1 = np.meshgrid(Xlow,Ylow)
# X2,Y2 = np.meshgrid(Xlow,Yhigh)
# X3,Y3 = np.meshgrid(Xhigh,Ylow)
# X4,Y4 = np.meshgrid(Xhigh,Yhigh)
#
# xn1, yn1 = X1.shape
# xn2, yn2 = X2.shape
# xn3, yn3 = X3.shape
# xn4, yn4 = X4.shape
#
# U1 = X1*0
# U2 = X2*0
# U3 = X3*0
# U4 = X4*4
#
# for xk in range(xn1):
#     for yk in range(yn1):
#         try:
#             z = complex(X1[xk,yk],Y1[xk,yk])
#             # w = np.angle(f(z))
#             # w = (f(z,0))
#             w = f(z)
#             if w != w:
#                 raise ValueError
#             U1[xk,yk] = w
#
#         except (ValueError, TypeError, ZeroDivisionError):
#             pass
# for xk in range(xn2):
#     for yk in range(yn2):
#         try:
#             z = complex(X2[xk,yk],Y2[xk,yk])
#             # w = np.angle(f(z))
#             # w = (f(z,0))
#             w = f(z)
#             if w != w:
#                 raise ValueError
#             U2[xk,yk] = w
#
#         except (ValueError, TypeError, ZeroDivisionError):
#             pass
# for xk in range(xn3):
#     for yk in range(yn3):
#         try:
#             z = complex(X3[xk,yk],Y3[xk,yk])
#             # w = np.angle(f(z))
#             # w = (f(z,0))
#             w = f(z)
#             if w != w:
#                 raise ValueError
#             U3[xk,yk] = w
#
#         except (ValueError, TypeError, ZeroDivisionError):
#             pass
# for xk in range(xn4):
#     for yk in range(yn4):
#         try:
#             z = complex(X4[xk,yk],Y4[xk,yk])
#             # w = np.angle(f(z))
#             # w = (f(z,0))
#             w = f(z)
#             if w != w:
#                 raise ValueError
#             U4[xk,yk] = w
#
#         except (ValueError, TypeError, ZeroDivisionError):
#             pass



X, Y = np.meshgrid(X, Y)
xn, yn = X.shape
W = X*0
for xk in range(xn):
    for yk in range(yn):
        try:
            z = complex(X[xk,yk],Y[xk,yk])
            # w = np.angle(f(z))
            # w = (f(z,0))
            w = f(z)
            if w != w:
                raise ValueError
            W[xk,yk] = w

        except (ValueError, TypeError, ZeroDivisionError):
            pass

# print(W1[0,0])
# print(W1[2,0])
# print(X[159])
# print(Y[159])
# print(W1[159])
# print(2*np.nan)
# Q= X*0
# for xk in range(xn):
#     for yk in range(yn):
#         try:
#             z = complex(X[xk,yk],Y[xk,yk])
#             # w = np.angle(f(z))
#             # w = (f(z,0))
#             w = g(z)
#             if w != w:
#                 raise ValueError
#             Q[xk,yk] = w
#
#         except (ValueError, TypeError, ZeroDivisionError):
#             pass
#
# J= X*0
# for xk in range(xn):
#     for yk in range(yn):
#         try:
#             z = complex(X[xk,yk],Y[xk,yk])
#             # w = np.angle(f(z))
#             # w = (f(z,0))
#             w = b(z)
#             if w != w:
#                 raise ValueError
#             J[xk,yk] = w
#
#         except (ValueError, TypeError, ZeroDivisionError):
#             pass
#
# K= X*0
# for xk in range(xn):
#     for yk in range(yn):
#         try:
#             z = complex(X[xk,yk],Y[xk,yk])
#             # w = np.angle(f(z))
#             # w = (f(z,0))
#             # if Y[yn] == 80:
#             #     w = np.nan
#             # else:
#             w = t(z)
#             # pos=np.where(np.abs(np.diff(w)) >= .5)[0]+1
#             # w = np.insert(w,pos, np.nan)
#             if w != w:
#                 raise ValueError
#             K[xk,yk] = w
#
#         except (ValueError, TypeError, ZeroDivisionError):
#             pass


# a = '$\eqn$'
# can comment out one of these
# L=[]
# for i in [x for x in np.arange(-1,1,.0125) if x != 80]:
#     L.append(i)

# pos = np.where(np.abs(np.diff(y)) >= 0.25)[0]+1
# x = np.insert(x, pos, np.nan)
# y = np.insert(y, pos, np.nan)



ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=cmap)
# ax.plot_surface(X,Y,Q,rstride=1,cstride=1,cmap=cmap)
# ax.plot_surface(X,Y,J,rstride=1,cstride=1,cmap=cmap)
# ax.plot_surface(X,Y,K,rstride=1,cstride=1,cmap=cmap)
# ax.plot_surface(X1, Y1, U1, rstride=1, cstride=1, cmap=cmap)
# ax.plot_surface(X2, Y2, U2, rstride=1, cstride=1, cmap=cmap)
# ax.plot_surface(X3, Y3, U3, rstride=1, cstride=1, cmap=cmap)
# ax.plot_surface(X4, Y4, U4, rstride=1, cstride=1, cmap=cmap)
# ax.plot_wireframe(X,Y,W,rstride=1, cstride=1)
# ax.set_xlabel('$Re[s_f]$')
# ax.set_ylabel('$Im[s_f]$')
ax.set_xlabel('$Re[z]$')
ax.set_ylabel('$Im[z]$')
# ax.set_zlabel('$Im[\mathcal{G}]$')
# ax.set_title(r'2nd sheet of $\mathcal{G}$, $s_i \neq s_f$')
# ax.set_zlabel('$Im[\mathcal{W} _ {df}]$')
# ax.set_title(r'1st sheet of $\mathcal{W} _ {df}$')
# ax.set_zlabel('$Im[z^{1/4}]$')
# ax.set_title(r'$z^{1/4}$')
ax.set_zlabel('$Re[func}]$')
ax.set_title(r'$Function in complex plane$')

# print("--- %s seconds ---" % (time.time() - start_time))

pylab.show()
