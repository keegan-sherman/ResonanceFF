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
# g = 3
mr = 2.2
xi = 1/2
ascale = .0001
# disp = 0
disp = .110117
pla = .110117
pi = 2.173 - .47915701951257783
# disp = 0.47915701951257783
# e = 0

ei = 1.05*(m1+m2)
# si = 0
# Q2 = 1.5
eps = 0
efrange = np.linspace(.05+disp*1j,3+disp*1j,1000)
refrange = np.linspace(.05,3,1000)
# Pz = 2*math.pi/15
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

def gamma(s,g):
    f = pow(g,2)/(6.0*cmath.pi)
    return f*pow(mr,2)*(qStar(s))/s

def tanBW(s,g):
    bottom = pow(mr,2)-s
    return np.sqrt(s)*gamma(s,g) / bottom

#this contains a psuedo point for plotting
def tanBWforplot(s):
    t = tanBW(s)
    if t > 3:
        return 3
    elif t == math.inf:
        return 3
    return t

def Kmatrix(s,g):
    top = 4*pow(g,2)*pow(mr,2)
    bottom = 3*xi*(pow(mr,2)-s)
    return top/bottom

#This is the real component M on the first sheet
def MI_real(s,g):
    a = np.real(1/((1/Kmatrix(s,g)) - (1j*rho(s))))
    b = np.imag(1/((1/Kmatrix(s,g)) - (1j*rho(s))))
    return a

#This is the imaginary component M on the first sheet
def MI_imag(s,g):
    a = np.real(1/((1/Kmatrix(s,g)) - (1j*rho(s))))
    b = np.imag(1/((1/Kmatrix(s,g)) - (1j*rho(s))))
    return b

#This is the real component M on the second sheet
def MII_real(s,g):
    a = np.real(1/((1/Kmatrix(s,g)) + (1j*rho(s))))
    b = np.imag(1/((1/Kmatrix(s,g)) + (1j*rho(s))))
    return a

#This is the imaginary component M on the second sheet
def MII_imag(s,g):
    a = np.real(1/((1/Kmatrix(s,g)) + (1j*rho(s))))
    b = np.imag(1/((1/Kmatrix(s,g)) + (1j*rho(s))))
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
def MI(sf,g):
    return MI_real(sf,g) +(1j*(MI_imag(sf,g)))

def MII(sf,g):
    return MII_real(sf,g) + (1j*(MII_imag(sf,g)))

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
    # b = np.subtract(Pf,Pi)
    # s = pf.dotprod(b,b)
    # Q2=-1*(s)

    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)

    Piterm = Pi*(GI(si,sf,Q2)-2*ComboG_b(si,sf,Q2))
    Pfterm = Pf*(GI(si,sf,Q2)-2*ComboG_a(si,sf,Q2))
    out = Piterm + Pfterm
    return out

def GvecII(Pi,Pf,Q2):
    # b = np.subtract(Pf,Pi)
    # s = pf.dotprod(b,b)
    # Q2=-1*(s)

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
    # b = np.subtract(Pf,Pi)
    # s = pf.dotprod(b,b)
    # Q2=-1*(s)
    si = pf.dotprod(Pi,Pi)
    sf = pf.dotprod(Pf,Pf)
    return MI(si)*f(Q2)*Gvec(Pi,Pf,Q2)*MI(sf)

def GII_term(Pi,Pf,Q2):
    # b = np.subtract(Pf,Pi)
    # s = pf.dotprod(b,b)
    # Q2=-1*(s)
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

def integrate_on_contour(function,path):
    result = 0.0
    for n in np.arange(len(path)-1):
        z0 = path[n]
        dz = path[n+1]-path[n]
        real_integrand = lambda x: np.real(function(z0+x*dz)*dz)
        imag_integrand = lambda x: np.imag(function(z0+x*dz)*dz)
        real_result = scipy.integrate.quad(real_integrand,0.0,1.0)[0]
        imag_result = scipy.integrate.quad(imag_integrand,0.0,1.0)[0]
        result += real_result + 1j*imag_result
    return result

def contourtest(z):
    return 3/(z-(4.05+.25j))

C1 = 4.721 - .48j
C2 = 4.722 - .48j
C3 = 4.722 - .479j
C4 = 4.721 - .479j
C5 = C1

B1 = 4.721 + .479j
B2 = 4.722 + .479j
B3 = 4.722 + .48j
B4 = 4.721 + .48j
B5 = B1

D1 = .2365 - .001*1j
D2 = .2365 + .001*1j
D3 = .2375 + .001*1j
D4 = .2375 - .001*1j
D5 = D1

EA1 = 2.15 - .111
EA2 = 2.18 - .111
EA3 = 2.18 - .109
EA4 = 2.15 - .109
EA5 = EA1

#g = .1
con1 = 4.838 - .0006j
con2 = 4.841 - .0006j
con3 = 4.841 - .0005j
con4 = 4.838 - .0005j
con5 = con1

#g = .5
tour1 = 4.838 - .014j
tour2 = 4.841 - .014j
tour3 = 4.841 - .012j
tour4 = 4.838 - .012j
tour5 = tour1

#g = 1
loop1 = 4.837 - .06j
loop2 = 4.84 - .06j
loop3 = 4.84 - .04j
loop4 = 4.837 - .04j
loop5 = loop1

#g = 1.5
ct1 = 4.83 - .13j
ct2 = 4.84 - .13j
ct3 = 4.84 - .11j
ct4 = 4.83 - .11j
ct5 = ct1

#g = 2
lt1 = 4.81 - .23j
lt2 = 4.82 - .23j
lt3 = 4.82 - .2j
lt4 = 4.81 - .2j
lt5 = lt1

#g = 2.5
r1 = 4.77 - .34j
r2 = 4.8 - .34j
r3 = 4.8 - .32j
r4 = 4.77 - .32j
r5 = r1

#g = 3.5
q1 = 4.61 - .655j
q2 = 4.62 - .655j
q3 = 4.62 - .645j
q4 = 4.61 - .645j
q5 = q1

#g = 4
w1 = 4.4 - .845j
w2 = 4.5 - .845j
w3 = 4.5 - .840j
w4 = 4.4 - .840j
w5 = w1

#g = 4.5
t1 = 4-1.1j
t2 = 4.1-1.1j
t3 = 4.1-1j
t4 = 4-1j
t5 = t1

#g = 5
k1 = 3.3 - 1.6j
k2 = 3.4 - 1.6j
k3 = 3.4 - 1.5j
k4 = 3.3 - 1.5j
k5 = k1

C = [C1,C2,C3,C4,C5]
BA = [B1,B2,B3,B4,B5]
D = [D5,D4,D3,D2,D1]
EA = [EA1,EA2,EA3,EA4,EA5]

con = [con1,con2,con3,con4,con5]
tour = [tour1,tour2,tour3,tour4,tour5]
loop = [loop1,loop2,loop3,loop4,loop5]
ct = [ct1,ct2,ct3,ct4,ct5]
lt = [lt1,lt2,lt3,lt4,lt5]
rval = [r1,r2,r3,r4,r5]
qval = [q1,q2,q3,q4,q5]
wval = [w1,w2,w3,w4,w5]
tval = [t1,t2,t3,t4,t5]
kval = [k1,k2,k3,k4,k5]

# listt = [con,tour,loop,ct,lt,rval,C,qval,wval,tval,kval]


data1swdfi = []
data1swdfr = []
data2swdfi = []
data2swdfr = []
srange = []
srangee = []

# print(efrange)
RElist = [1.5,2.5]


data11 = []
data12 = []
data13 = []
data14 = []
data15 = []
data16 = []
data17 = []
data18 = []
data19 = []
data110 = []
data111 = []
data21 = []
data22 = []
data23 = []
data24 = []
data25 = []
data26 = []
data27 = []
data28 = []
data29 = []
data210 = []
data211 = []

def Pfz(efs,Q2):
    return (efs-np.real(pi))**2 - Q2

q2range = np.linspace(0,4,1000)

# for ef in refrange:
#     for Q2s in q2range:
#         P1 = np.array([2.173-.110117j,0,0,0])
#
#         Pf = Pfz(ef,Q2s)
#         P2 = np.array([ef-.110117,0,0,Pf])
#
#         si = pf.dotprod(P1,P1)
#         sf = pf.dotprod(P2,P2)
#
#         G = GII(si,sf,Q2s)
#         f = f(Q2s)
#         A = AparamII(P1,P2,Q2s)
#
#         residue = integral1/(2*math.pi*1j)
#         reduce = residue**2
#
#         frr = reduce*(f*G+A)
#
#         data1.append(np.real(frr))
#         data2.append(np.imag(frr))

real = []
imag = []
gval = [0.1,.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
listt = [con,tour,loop,ct,lt,rval,C,qval,wval,tval,kval]

for i,g in enumerate(gval):
    lyst = listt[i]

    integral = integrate_on_contour(lambda z: MII(z,g),lyst)
    residue = integral/(2*math.pi*1j)

    for Q2s in q2range:
        sr = 4.7215 - .4794j
        scale = 0.1

        G = GII(sr,sr,Q2s)
        fr = f(Q2s)
        Ar = scale*fr/g**2

        frr = (-residue*(fr*G+Ar))

        real.append(np.real(frr))
        imag.append(np.imag(frr))

for h in range(11000):
    if h < 1000:
        data11.append(real[h])
        data21.append(imag[h])
    elif h >= 1000 and h < 2000:
        data12.append(real[h])
        data22.append(imag[h])
    elif h >= 2000 and h < 3000:
        data13.append(real[h])
        data23.append(imag[h])
    elif h >= 3000 and h < 4000:
        data14.append(real[h])
        data24.append(imag[h])
    elif h >= 4000 and h < 5000:
        data15.append(real[h])
        data25.append(imag[h])
    elif h >= 5000 and h < 6000:
        data16.append(real[h])
        data26.append(imag[h])
    elif h >= 6000 and h < 7000:
        data17.append(real[h])
        data27.append(imag[h])
    elif h >= 7000 and h < 8000:
        data18.append(real[h])
        data28.append(imag[h])
    elif h >= 8000 and h < 9000:
        data19.append(real[h])
        data29.append(imag[h])
    elif h >= 9000 and h < 10000:
        data110.append(real[h])
        data210.append(imag[h])
    elif h >= 10000:
        data111.append(real[h])
        data211.append(imag[h])

# for Q2s in q2range:
#     sr = 4.7215 - .4794j
#     scale = 0.1
#     # si = sr
#     # sf = sr
#
#     G = GII(sr,sr,Q2s)
#     fr = f(Q2s)
#     Ar1 = scale*fr/gval[0]**2
#     Ar2 = scale*fr/gval[1]**2
#     Ar3 = scale*fr/gval[2]**2
#     Ar4 = scale*fr/gval[3]**2
#     Ar5 = scale*fr/gval[4]**2
#     Ar6 = scale*fr/gval[5]**2
#     Ar7 = scale*fr/gval[6]**2
#
#     integral1 = integrate_on_contour(lambda z: MII(z,gval[0]),con)
#     integral2 = integrate_on_contour(lambda z: MII(z,gval[1]),tour)
#     integral3 = integrate_on_contour(lambda z: MII(z,gval[2]),loop)
#     integral4 = integrate_on_contour(lambda z: MII(z,gval[3]),ct)
#     integral5 = integrate_on_contour(lambda z: MII(z,gval[4]),lt)
#     integral6 = integrate_on_contour(lambda z: MII(z,gval[5]),rval)
#     integral7 = integrate_on_contour(lambda z: MII(z,gval[6]),C)
#
#     residue1 = integral1/(2*math.pi*1j)
#     residue2 = integral2/(2*math.pi*1j)
#     residue3 = integral3/(2*math.pi*1j)
#     residue4 = integral4/(2*math.pi*1j)
#     residue5 = integral5/(2*math.pi*1j)
#     residue6 = integral6/(2*math.pi*1j)
#     residue7 = integral7/(2*math.pi*1j)
#
#     frr1 = -residue1*(fr*G+Ar1)
#     frr2 = -residue2*(fr*G+Ar2)
#     frr3 = -residue3*(fr*G+Ar3)
#     frr4 = -residue4*(fr*G+Ar4)
#     frr5 = -residue5*(fr*G+Ar5)
#     frr6 = -residue6*(fr*G+Ar6)
#     frr7 = -residue7*(fr*G+Ar7)
#
#     data11.append(np.real(frr1))
#     data12.append(np.real(frr2))
#     data13.append(np.real(frr3))
#     data14.append(np.real(frr4))
#     data15.append(np.real(frr5))
#     data16.append(np.real(frr6))
#     data17.append(np.real(frr7))
#
#     data21.append(np.imag(frr1))
#     data22.append(np.imag(frr2))
#     data23.append(np.imag(frr3))
#     data24.append(np.imag(frr4))
#     data25.append(np.imag(frr5))
#     data26.append(np.imag(frr6))
#     data27.append(np.imag(frr7))






mpl.rcParams['mathtext.rm'] = 'Yrsa'
mpl.rcParams['mathtext.it'] = 'Yrsa:italic'
mpl.rcParams['mathtext.bf'] = 'Yrsa:bold'

# fig, plots2 = plt.subplots(2,1)
# plots2[0].axhline(0, color = 'k')
# plots2[1].axhline(0, color = 'k')
# plots2[0].plot(q2range,data11, color = 'darkgreen',label = 'g = .1')
# plots2[0].plot(q2range,data12, color = 'green',label = 'g = .5')
# plots2[0].plot(q2range,data13, color = 'forestgreen',label = 'g = 1')
# plots2[0].plot(q2range,data14, color = 'limegreen',label = 'g = 1.5')
# plots2[0].plot(q2range,data15, color = 'lime',label = 'g = 2')
# plots2[0].plot(q2range,data16, color = 'blue',label = 'g = 2.5')
# plots2[0].plot(q2range,data17, color = 'lightblue',label = 'g = 3')
# plots2[1].plot(q2range,data21, color = 'hotpink',label = 'g =.1')
# plots2[1].plot(q2range,data22, color = 'deeppink', label = 'g= .5')
# plots2[1].plot(q2range,data23, color = 'magenta',label = 'g = 1')
# plots2[1].plot(q2range,data24, color = 'fuchsia',label = 'g = 1.5')
# plots2[1].plot(q2range,data25, color = 'darkmagenta',label = 'g = 2')
# plots2[1].plot(q2range,data26, color = 'red',label = 'g = 2.5')
# plots2[1].plot(q2range,data27, color = 'lightred',label = 'g = 3')
# plots2[0].legend()
# plots2[1].legend()
# plots2[0].set_title(r'f')
# plots2[1].set(xlabel='$Q^2$')
# plots2[0].set_ylabel(ylabel='$Re[f]$',fontname="Yrsa",size=10)
# plots2[1].set_ylabel(ylabel='$Im[f]$',fontname="Yrsa",size=10)
#
# plt.show()

# plt.axhline(0, color = 'k')
# plt.axhline(0, color = 'k')
# plt.plot(q2range,data11, color = 'aquamarine',label = 'g = .1')
# plt.plot(q2range,data12, color = 'springgreen',label = 'g = .5')
# plt.plot(q2range,data13, color = 'lime',label = 'g = 1')
# plt.plot(q2range,data14, color = 'limegreen',label = 'g = 1.5')
# plt.plot(q2range,data15, color = 'forestgreen',label = 'g = 2')
# plt.plot(q2range,data16, color = 'green',label = 'g = 2.5')
# plt.plot(q2range,data17, color = 'darkgreen',label = 'g = 3')
# plt.plot(q2range,data18, color = 'darkgreen',label = 'g = 3.5')
# plt.plot(q2range,data19, color = 'darkgreen',label = 'g = 4')
# plt.plot(q2range,data110, color = 'darkgreen',label = 'g = 4.5')
# plt.plot(q2range,data111, color = 'darkgreen',label = 'g = 5')
# plt.legend()
# plt.title(r'$f_{R {\rightarrow} R}$')
# plt.xlabel('$Q^2$')
# plt.ylabel(r'$Re[f_{R {\rightarrow} R}]$',fontname="Yrsa",size=10)
# plt.show()

plt.axhline(0, color = 'k')
plt.axhline(0, color = 'k')
plt.plot(q2range,data21, color = 'hotpink',label = 'g = .1')
plt.plot(q2range,data22, color = 'deeppink',label = 'g = .5')
plt.plot(q2range,data23, color = 'magenta',label = 'g = 1')
plt.plot(q2range,data24, color = 'fuchsia',label = 'g = 1.5')
plt.plot(q2range,data25, color = 'darkmagenta',label = 'g = 2')
plt.plot(q2range,data26, color = 'red',label = 'g = 2.5')
plt.plot(q2range,data27, color = 'lightcoral',label = 'g = 3')
plt.plot(q2range,data28, color = 'crimson',label = 'g = 3.5')
plt.plot(q2range,data29, color = 'darkred',label = 'g = 4')
plt.plot(q2range,data210, color = 'purple',label = 'g = 4.5')
plt.plot(q2range,data211, color = 'violet',label = 'g = 5')
plt.legend()
plt.title(r'$f_{R {\rightarrow} R}$')
plt.xlabel('$Q^2$')
plt.ylabel(r'$Im[f_{R {\rightarrow} R}]$',fontname="Yrsa",size=10)
plt.show()
