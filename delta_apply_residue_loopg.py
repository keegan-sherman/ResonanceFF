# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import pylab
import numpy as np
import mpmath
import physfunc as pf
import math
import cmath
# from Pole_Finder_for_BW_class import BWPoleFinder
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

def quadraticSolution(a,b,c):
	sqrtTerm = math.sqrt(pow(b,2)-4*a*c)
	point1 = (-b-sqrtTerm)/(2*a)
	point2 = (-b+sqrtTerm)/(2*a)
	return point1, point2

def crits(m1,m2,Q2,si,sf):
	points = []

	# Critical points for A^{2} = -B
	alpha = -1*(Q2+sf+si)/si
	beta = 1 + (pow(m2,2)-pow(m1,2))/si
	gamma = 4*pow(m2,2)/si
	eta = 4*(pow(m2,2)-pow(m1,2))/si
	phi = 4*sf/si
	a = pow(alpha,2) - phi
	b = 2*alpha*beta + eta + phi
	c = pow(beta,2) - gamma
	if pow(b,2) >= 4*a*c and a != 0:
		point1, point2 = quadraticSolution(a,b,c)
		if point1 > 0 and point1 < 1:
			points.append(point1)
		if point2 > 0 and point1 < 1:
			points.append(point2)

	# Critical points for B = 0
	if sf >= pow(m1+m2,2):
		b = sf + pow(m2,2) - pow(m1,2)
		point1, point2 = quadraticSolution(sf,-b,pow(m2,2))
		if point1 > 0 and point1 < 1:
			points.append(point1)
		if point2 > 0 and point2 < 1:
			points.append(point2)

	return points


#####################
#this is the TRUE M
def MI(sf,g):
    return MI_real(sf,g) +(1j*(MI_imag(sf,g)))

def MII(sf,g):
    return MII_real(sf,g) + (1j*(MII_imag(sf,g)))

def IaFr(si,sf,Q2):
    crit = np.real(crits(m1,m2,Q2.real,si.real,sf.real))
    IaR,error = scipy.integrate.quad(realF,0,1,args=(si,sf,Q2),points=crit)
    # IaR,error = scipy.integrate.quad(realF,0,1,args=(si,sf,Q2))
    return (IaR)

def IaFi(si,sf,Q2):
    crit = np.real(crits(m1,m2,Q2.real,si.real,sf.real))
    IaI,error2 = scipy.integrate.quad(imagF,0,1,args=(si,sf,Q2),points=crit)
    # IaI,error2 = scipy.integrate.quad(imagF,0,1,args=(si,sf,Q2))
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

data1swdfi = []
data1swdfr = []
data2swdfi = []
data2swdfr = []
srange = []
srangee = []

RElist = [1.5,2.5]


def Pfz(efs,Q2):
    return (efs-np.real(pi))**2 - Q2

q2range = np.linspace(0,4,1000)

real = []
imag = []
abs = []
gval = [0.1,.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
listt = [con,tour,loop,ct,lt,rval,C,qval,wval,tval,kval]
roots = []

for gv in gval:
    a = 1
    b = -(2*mr**2)
    c = (mr**4 + (mr**4*gv**4)/(144*(math.pi)**2))
    d = -(mr**4*gv**4/(36*(math.pi)**2))
    coeff = [a,b,c,d]
    roots.append(np.roots(coeff)[1])

# realPole = {}
# imagPole = {}
# absPole = {}
#
# for i,g in enumerate(gval):
#     lyst = listt[i]
#     sr = roots[i]
#
#     integral = integrate_on_contour(lambda z: MII(z,g),lyst)
#     residue = integral/(2*math.pi*1j)
#
#     for Q2s in q2range:
#         # sr = 4.7215 - .4794j
#         scale = 0.1
#
#         G = GII(sr,sr,Q2s)
#         fr = f(Q2s)
#         Ar = scale*fr/g**2
#
#         frr = (-residue*(fr*G+Ar))
#
#         real.append(np.real(frr))
#         imag.append(np.imag(frr))
#         abs.append(np.abs(frr))
#
#     realPole[g]=real
#     imagPole[g]=imag
#     absPole[g] = abs
#     real = []
#     imag = []
#     abs = []
#
s_p = [4.839169169169169, 4.839169169169169, 4.839169169169169, 4.834174174174174, 4.819189189189189, 4.784224224224224, 4.719289289289289, 4.6093993993994, 4.404604604604605, 4.01, 4.01]
#
# real1 = []
# imag1 = []
# abs1 = []
#
# realPeak = {}
# imagPeak = {}
# absPeak ={}
# ratioreal = {}
# ratioimag = {}
# ratioabs = {}
#
# for i,g in enumerate(gval):
#     lyst = listt[i]
#     sp = s_p[i]
#
#     # integral = integrate_on_contour(lambda z: MII(z,g),lyst)
#     # residue = integral/(2*math.pi*1j)
#
#     for Q2s in q2range:
#         # sr = 4.7215 - .4794j
#         scale = 0.1
#
#         # G = GII(sp,sp,Q2s)
#         G = GI(sp,sp,Q2s)
#         fr = f(Q2s)
#         Ar = scale*fr/g**2
#
#         c = 2*g*mr/(np.sqrt(3*xi))
#
#
#         frr = (c**2*(fr*G+Ar))
#
#         real1.append(np.real(frr))
#         imag1.append(np.imag(frr))
#         abs1.append(np.abs(frr))
#
#     realPeak[g]=real1
#     imagPeak[g]=imag1
#     absPeak[g]=abs1
#     real1 = []
#     imag1 = []
#     abs1=[]
#
# allrpo = []
# allrpe = []
# allipo = []
# allipe = []
# allapo = []
# allape = []
# rr = []
# ir = []
# ar = []
#
# for i,g in enumerate(gval):
#     for j,b in enumerate(range(1000)):
#         allrpo.append(realPole[g][j])
#         allrpe.append(realPeak[g][j])
#         allipo.append(imagPole[g][j])
#         allipe.append(imagPeak[g][j])
#         allapo.append(absPole[g][j])
#         allapo.append(absPeak[g][j])
#         rr.append((realPole[g][j])/(realPeak[g][j]))
#         ir.append((imagPole[g][j])/(imagPeak[g][j]))
#         ar.append((absPeak[g][j])/(absPole[g][j]))
#     ratioreal[g]=rr
#     ratioimag[g]=ir
#     ratioabs[g]=ar
#     rr = []
#     ir = []
#     ar = []

# Here we begin looking at the error plots
#
#
#
absq = []
absPoleq20 = {}

qval = 0

for i,g in enumerate(gval):
    lyst = listt[i]
    sr = roots[i]

    integral = integrate_on_contour(lambda z: MII(z,g),lyst)
    residue = integral/(2*math.pi*1j)

    scale = 0.1

    G = GII(sr,sr,qval)
    fr = f(qval)
    Ar = scale*fr/g**2

    frr = (-residue*(fr*G+Ar))

    absq.append(np.abs(frr))

    absPoleq20[g] = absq
    absq = []

absq1 = []

absPeakq20 ={}
ratioabsq20 = {}
for i,g in enumerate(gval):
    lyst = listt[i]
    sp = s_p[i]

    # integral = integrate_on_contour(lambda z: MII(z,g),lyst)
    # residue = integral/(2*math.pi*1j)

    scale = 0.1

    # G = GII(sp,sp,Q2s)
    G = GI(sp,sp,qval)
    fr = f(qval)
    Ar = scale*fr/g**2

    c = 2*g*mr/(np.sqrt(3*xi))


    frr = (c**2*(fr*G+Ar))

    absq1.append(np.abs(frr))

    absPeakq20[g]=absq1
    absq1=[]

arq = []

for i,g in enumerate(gval):
    arq.append((absPeakq20[g][0])/(absPoleq20[g][0]))
    ratioabsq20[g]=arq
    arq = []

dw=[]
sigma=[]
for i in range(7):
    dw.append(-2*np.imag(csqrt(roots[i]))/np.real(csqrt(roots[i])))

gval2 = [0.1,.5,1,1.5,2,2.5,3]

for i,g in enumerate(gval2):
    sigma.append(np.abs((ratioabsq20[g][0]-1)*100))



print(dw)
# print(absPoleq20[2][0])
print(sigma)

mpl.rcParams['mathtext.rm'] = 'Yrsa'
mpl.rcParams['mathtext.it'] = 'Yrsa:italic'
mpl.rcParams['mathtext.bf'] = 'Yrsa:bold'

plt.plot(dw,sigma, color = 'darkgreen',label = 'g = .1')

# plt.plot(q2range,ratioabs[.1], color = 'red',label = 'g = .1')
# plt.plot(q2range,ratioabs[.5], color = 'orange',label = 'g = .5')
# plt.plot(q2range,ratioabs[1], color = 'yellow',label = 'g = 1')
# plt.plot(q2range,ratioabs[1.5], color = 'lawngreen',label = 'g = 1.5')
# plt.plot(q2range,ratioabs[2], color = 'darkgreen',label = 'g = 2')
# plt.plot(q2range,ratioabs[2.5], color = 'cyan',label = 'g = 2.5')
# plt.plot(q2range,ratioabs[3], color = 'blue',label = 'g = 3')
# plt.axhline(0, color = 'k')
# plt.legend()
# plt.title(r'$Ratio Peak over Pole as a function of Q^2$')
# plt.xlabel('$Q^2$')
# plt.ylabel(r'$|f_{R {\rightarrow} R, {peak}}/f_{R {\rightarrow} R, {pole}}|$',fontname="Yrsa",size=10)
plt.title("Sigma plots")
plt.xlabel(r'$\Gamma_R/m_r$')
plt.ylabel(r'$\sigma (\%)$',fontname="Yrsa",size=10)
plt.show()
