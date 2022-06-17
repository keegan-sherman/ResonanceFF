import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import quad
import sympy

#Greg Blume

#these are the f(x) functions used in the L functions
#the fucntions with the extra s on the end correspond to an epsilon value greater that 0
#they are used to find the sign of yplus and yminus so that we can use the limit sympy.limit(Imy,eps,0)= cmath.pi/2
#then from that limit we can get the approximation

def dotprod(a,b):
    q = a[0]*b[0]
    w = a[1]*b[1]
    e = a[2]*b[2]
    r = a[3]*b[3]
    return q - w - e - r

def bisectionalapprox(f,a,b,N):
    #this means its range is to large
    if f(a)*f(b) >= 0:
        return None

    #initialize the values to update
    a_update = a
    b_update = b
    # iteration = 0

    for i,iteration in enumerate(range(1,N+1)):
        #here we introduce an update parameter
        #new is the new term that will "update" a and b
        new_bound = (a_update + b_update)/2
        f_update = f(new_bound)

        #new right bound
        if f(a_update)*f_update < 0:
            a_update = a_update
            b_update = new_bound
            # iteration += 1

        #new left bound
        elif f(b_update)*f_update < 0:
            a_update = new_bound
            b_update = b_update
            # iteration += 1

        #solved
        elif f_update == 0:
            # iteration += 1
            return new_bound , iteration

        #incase value not found in number of interations
        else:
            return None

    return (a_update + b_update)/2 , iteration

def Newton(f, dfdx, x, eps):
    function = f(x)
    #initialize iteration counter
    iteration = 0

    while abs(function) > eps and iteration < 100:

        try:
            x = x - float(function)/dfdx(x)

        except ZeroDivisionError:
            print("derivative zero")

        function = f(x)
        iteration += 1

    # Here too many iterations
    if abs(function) > eps:
        iteration = -1

    return x, iteration

def csqrt(z):
    arg = np.angle(z)
    norm = np.absolute(z)
    newarg = np.mod(arg + 2*math.pi, 2*math.pi) - 2*math.pi
    return np.sqrt(norm)*np.exp(1j*newarg/2)

def dbydx(inputfunction,x,dx,args=()):
    top = inputfunction(x+dx,*args) - inputfunction(x-dx,*args)
    bottom = 2*dx
    return top/bottom

def A(x,m1,m2,Q2,eistar,efstar,eps):
    sth = pow((m1+m2),2)
    si = eistar**2
    sf = (efstar**2)
    num1 = (pow(m2,2)-pow(m1,2)-(x*(Q2+sf+si)))
    frac1 = num1/si
    return 1 + frac1

def B(x,m1,m2,Q2,eistar,efstar,eps):
    sth = pow((m1+m2),2)
    si = eistar**2
    sf = (efstar**2)
    num2 = (pow(m2,2)-((x*((pow(m2,2))-pow(m1,2))))-(x*(1-x)*sf))
    frac2 = num2/si
    return (-4) * frac2

def yplus(x,m1,m2,Q2,eistar,efstar,eps):
    sqrt = cmath.sqrt(A(x,m1,m2,Q2,eistar,efstar,eps)**2+B(x,m1,m2,Q2,eistar,efstar,eps)+(1j*eps))
    return .5 * (A(x,m1,m2,Q2,eistar,efstar,eps)+sqrt)

def yminus(x,m1,m2,Q2,eistar,efstar,eps):
    sqrt = cmath.sqrt(A(x,m1,m2,Q2,eistar,efstar,eps)**2+B(x,m1,m2,Q2,eistar,efstar,eps)+(1j*eps))
    return .5 * (A(x,m1,m2,Q2,eistar,efstar,eps)-sqrt)


#here is the Lplus and Lminus equations, within them there is an if statement that allows the computer to use the
#proper sign of the limit depending on the sign above the above function
#This if statement only occurs with Imy or Imyb equals 0 thus causing one to divide by 0 when eps = 0

def Lplus(x,m1,m2,Q2,eistar,efstar,eps):
    ins = (1-x-yplus(x,m1,m2,Q2,eistar,efstar,eps))/(yplus(x,m1,m2,Q2,eistar,efstar,eps))
    abs = np.abs(ins)
    log = cmath.log(abs)
    Rey = np.real(yplus(x,m1,m2,Q2,eistar,efstar,eps))
    Imy = np.imag(yplus(x,m1,m2,Q2,eistar,efstar,eps))

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

def Lminus(x,m1,m2,Q2,eistar,efstar,eps):
    abs2 = np.abs((1-x-yminus(x,m1,m2,Q2,eistar,efstar,eps))/(yminus(x,m1,m2,Q2,eistar,efstar,eps)))
    log2 = np.log(abs2)
    Reyb = np.real(yminus(x,m1,m2,Q2,eistar,efstar,eps))
    Imyb = np.imag(yminus(x,m1,m2,Q2,eistar,efstar,eps))

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

#here is the general f(x), f1 and f2 as well, equations that will be integrated

def F(x,m1,m2,Q2,eistar,efstar,eps):
    sth = pow((m1+m2),2)
    si = eistar**2
    fpi = 4*cmath.pi
    fpisq = pow(fpi,2)
    lead = 1 / (fpisq*si)
    numf = (Lplus(x,m1,m2,Q2,eistar,efstar,eps)-Lminus(x,m1,m2,Q2,eistar,efstar,eps))
    denf = (yplus(x,m1,m2,Q2,eistar,efstar,eps)-yminus(x,m1,m2,Q2,eistar,efstar,eps))
    fracf = numf/denf
    return lead * fracf

def F11(x,m1,m2,Q2,eistar,efstar,eps):
    return x * F(x,m1,m2,Q2,eistar,efstar,eps)

def F12(x,m1,m2,Q2,eistar,efstar,eps):
    sth = pow((m1+m2),2)
    si = eistar**2
    fpi = 4*cmath.pi
    fpisq = pow(fpi,2)
    lead = 1 / (fpisq*si)
    numf = ((yplus(x,m1,m2,Q2,eistar,efstar,eps)*Lplus(x,m1,m2,Q2,eistar,efstar,eps))-(yminus(x,m1,m2,Q2,eistar,efstar,eps)*Lminus(x,m1,m2,Q2,eistar,efstar,eps)))
    denf = (yplus(x,m1,m2,Q2,eistar,efstar,eps)-yminus(x,m1,m2,Q2,eistar,efstar,eps))
    fracf = numf/denf
    return lead * fracf

def Ref(x,m1,m2,Q2,eistar,efstar,eps):
    return np.real(F(x,m1,m2,Q2,eistar,efstar,eps))

def Ref11(x,m1,m2,Q2,eistar,efstar,eps):
    return np.real(F11(x,m1,m2,Q2,eistar,efstar,eps))

def Ref12(x,m1,m2,Q2,eistar,efstar,eps):
    return np.real(F12(x,m1,m2,Q2,eistar,efstar,eps))

def Imf(x,m1,m2,Q2,eistar,efstar,eps):
    return np.imag(F(x,m1,m2,Q2,eistar,efstar,eps))

def Imf11(x,m1,m2,Q2,eistar,efstar,eps):
    return np.imag(F11(x,m1,m2,Q2,eistar,efstar,eps))

def Imf12(x,m1,m2,Q2,eistar,efstar,eps):
    return np.imag(F12(x,m1,m2,Q2,eistar,efstar,eps))

#here we define a function that locates the critical points and puts them in a list for us to tell the cmputer to
#be aware of when integrating

def crits(m1,m2,Q2,eistar,efstar,eps):
    points = []
    sf = (efstar**2)
    sth = pow((m1+m2),2)
    si = eistar**2
    l = (Q2+sf+si)

    A1 = pow((-1*(Q2+sf+si)/si),2) - (4*sf/si)
    B1 = 2*(-1*(Q2+sf+si)/si)*(1 + (pow(m2,2)-pow(m1,2))/si) + (4*(pow(m2,2)-pow(m1,2))/si) + (4*sf/si)
    C1 = pow((1 + (pow(m2,2)-pow(m1,2))/si),2) - (4*(pow(m2,2))/si)


    if pow(B1,2) >= 4*A1*C1 and A1 != 0:
        point1 = (((-B1 + math.sqrt((B1**2)-4*A1*C1))/(2*A1)))
        point2 = (((-B1 - math.sqrt((B1**2)-4*A1*C1))/(2*A1)))
        # point1 = (((-B1 + csqrt((B1**2)-4*A1*C1))/(2*A1)))
        # point2 = (((-B1 - csqrt((B1**2)-4*A1*C1))/(2*A1)))
        if point1 > 0 and point1 < 1:
            points.append(point1)
        if point2 > 0 and point2 < 1:
            points.append(point2)

    A2 = sf
    B2 = ((m1**2)-(m2**2)-sf)
    C2 = (m2**2)

    if sf >= sth:
        points.append(((-B2 + math.sqrt((B2**2)-4*A2*C2))/(2*A2)))
        points.append(((-B2 - math.sqrt((B2**2)-4*A2*C2))/(2*A2)))
        # points.append(((-B2 + csqrt((B2**2)-4*A2*C2))/(2*A2)))
        # points.append(((-B2 - csqrt((B2**2)-4*A2*C2))/(2*A2)))

    return points

#here we integrate either the strictly real or stricly imag equations aboven= the critical function

def IaF(m1,m2,Q2,eistar,efstar,eps):
    crit = (crits(m1,m2,Q2,eistar,efstar,eps))
    IaR,error = scipy.integrate.quad(Ref,0,1,args=(m1,m2,Q2,eistar,efstar,eps),points=crit)
    IaI,error2 = scipy.integrate.quad(Imf,0,1,args=(m1,m2,Q2,eistar,efstar,eps),points=crit)
    return np.complex(IaR,IaI)

def IaF11(m1,m2,Q2,eistar,efstar,eps):
    crit = (crits(m1,m2,Q2,eistar,efstar,eps))
    IaR,error = scipy.integrate.quad(Ref11,0,1,args=(m1,m2,Q2,eistar,efstar,eps),points=crit)
    IaI,error2 = scipy.integrate.quad(Imf11,0,1,args=(m1,m2,Q2,eistar,efstar,eps),points=crit)
    return np.complex(IaR,IaI)

def IaF12(m1,m2,Q2,eistar,efstar,eps):
    crit = (crits(m1,m2,Q2,eistar,efstar,eps))
    IaR,error = scipy.integrate.quad(Ref12,0,1,args=(m1,m2,Q2,eistar,efstar,eps),points=crit)
    IaI,error2 = scipy.integrate.quad(Imf12,0,1,args=(m1,m2,Q2,eistar,efstar,eps),points=crit)
    return np.complex(IaR,IaI)

def Gvec(m1,m2,Pi,Pf,eps):
    si = dotprod(Pi,Pi)
    sf = dotprod(Pf,Pf)
    eistar = math.sqrt(si)
    efstar = math.sqrt(sf)
    b = np.subtract(Pf,Pi)
    s = dotprod(b,b)
    Q2=-1*(s)
    Piterm = Pi*(IaF(m1,m2,Q2,eistar,efstar,eps)-2*IaF12(m1,m2,Q2,eistar,efstar,eps))
    Pfterm = Pf*(IaF(m1,m2,Q2,eistar,efstar,eps)-2*IaF11(m1,m2,Q2,eistar,efstar,eps))
    out = Piterm + Pfterm
    return out

def I_nu(m1,m2,Pi,Pf,eps):
    si = dotprod(Pi,Pi)
    sf = dotprod(Pf,Pf)
    eistar = math.sqrt(si)
    efstar = math.sqrt(sf)
    b = np.subtract(Pf,Pi)
    s = dotprod(b,b)
    Q2=-1*(s)
    out = Pf*IaF11(m1,m2,Q2,eistar,efstar,eps)+Pi*IaF12(m1,m2,Q2,eistar,efstar,eps)
    return out

def qStar(m1,m2,s):
	return 1/2 * csqrt((s - 2*((m1**2)+(m2**2)) + ((((m2**2)-(m1**2))**2)/s)))

def gamma(m1,m2,g,s,mr):
    f = pow(g,2)/(6.0*cmath.pi)
    return f*pow(mr,2)*(qStar(m1,m2,s))/s

def tanBW(m1,m2,g,s,mr):
    bottom = pow(mr,2)-s
    return np.sqrt(s)*gamma(m1,m2,g,s,mr) / bottom

def M(m1,m2,g,s,mr,xi):
	qcot = qStar(m1,m2,s)/tanBW(m1,m2,g,s,mr)
	iq = (1j)*qStar(m1,m2,s)
	bottom = qcot - iq
	top = (8*math.pi*np.sqrt(s))/ xi
	return top / bottom

def M_1(s,m1,m2,g,mr,xi):
	qcot = qStar(m1,m2,s)/tanBW(m1,m2,g,s,mr)
	iq = (1j)*qStar(m1,m2,s)
	bottom = qcot - iq
	top = (8*math.pi*csqrt(s))/ xi
	return top / bottom

def dbydxofM(m1,m2,g,s,mr,xi):
    numlead = 48*pow(cmath.pi,2)*s/(pow(g,2)*pow(mr,2))
    nummid = (1j)*qStar(m1,m2,s)*4*cmath.pi/csqrt(s)
    numfin = (1j)*cmath.pi*csqrt(s)*(1-(((m2**2)-(m1**2))/pow(s,2)))/qStar(m1,m2,s)
    num = numlead - nummid + numfin
    den = xi*(((pow(mr,2)-s)*6*cmath.pi*csqrt(s)/(pow(g,2)*pow(mr,2)))-((1j)*qStar(m1,m2,s)))**2
    return num/den


def dbydxofM2(m1,m2,g,s,mr,xi,ds):
    dbydxofM = dbydx(M_1,s,ds,(m1,m2,g,mr,xi))
    return dbydxofM

def f(Q2,mr):
    bot = mr**2 + Q2
    return mr**2 / bot

def caliF(Q2,mr):
    scale = .001
    return scale * f(Q2,mr)

def Wdf(m1,m2,mr,Q2,sf,si,eps):
    eistar = csqrt(si)
    efstar = csqrt(sf)
    first = M_1(sf,m1,m2,g,mr)*caliF(Q2,mr)*M_1(si,m1,m2,g,mr)
    second = M_1(sf,m1,m2,g,mr)*f(Q2,mr)*IaF(m1,m2,Q2,eistar,efstar,eps)*M_1(si,m1,m2,g,mr)
    return (first + second)

#this uses my manual derivative
#if you want to check values use array component here
def Wvector(P_input,m1,m2,mr,g,xi):
    Q = 1
    s = dotprod(P_input,P_input)
    return 2*P_input*Q*dbydxofM(m1,m2,g,s,mr,xi)


#this uses the computers derivative
def Wvector2(P_input,m1,m2,mr,g,xi,ds):
    Q = 1
    s = dotprod(P_input,P_input)
    dbydxofM = dbydx(M_1,s,ds,(m1,m2,g,mr,xi))
    return 2*P_input*Q*dbydxofM

def Fvector(P_input,m1,m2,mr,g,xi):
    Q = 1
    s = dotprod(P_input,P_input)
    front = (1/M(m1,m2,g,s,mr,xi))*Wvector(P_input,m1,m2,mr,g,xi)*(1/M(m1,m2,g,s,mr,xi))
    back = f(0,mr)*Gvec(m1,m2,P_input,P_input,0)
    return front - back

def Fvector2(P_input,m1,m2,mr,g,xi,ds):
    Q = 1
    s = dotprod(P_input,P_input)
    front = (1/M(m1,m2,g,s,mr,xi))*Wvector2(P_input,m1,m2,mr,g,xi,ds)*(1/M(m1,m2,g,s,mr,xi))
    back = f(0,mr)*Gvec(m1,m2,P_input,P_input,0)
    return front - back

def Gveccutkowski(P_input,m1,m2):
    s = dotprod(P_input,P_input)
    lead = P_input*1j*math.pi/((4*math.pi)**2*pow(s,3/2)*qStar(m1,m2,s))
    follow = ((m1**2-m2**2)**2)/s - ((m1**2+m2**2))
    imag = lead*follow
    return imag

#this is the cutkowski F
def Fvector3(P_input,m1,m2,mr,g,xi):
    Q = 1
    s = dotprod(P_input,P_input)
    front = (1/M(m1,m2,g,s,mr,xi))*Wvector(P_input,m1,m2,mr,g,xi)*(1/M(m1,m2,g,s,mr,xi))
    back = f(0,mr)*Gveccutkowski(P_input,m1,m2)
    return front - back

def Fpole(P_input,m1,m2,mr,g,xi,ds,que,mp):
    bot = mp**2 + que
    return (mp**2 / bot)*Fvector2(P_input,m1,m2,mr,g,xi,ds)

def F_0(P_input,m1,m2,mr,g,xi,ds,mp):
    que=0
    bot = mp**2 + que
    return (mp**2 / bot)*Fvector2(P_input,m1,m2,mr,g,xi,ds)

def F_15(P_input,m1,m2,mr,g,xi,ds,mp):
    que=1.5
    bot = mp**2 + que
    return (mp**2 / bot)*Fvector2(P_input,m1,m2,mr,g,xi,ds)

def F_100(P_input,m1,m2,mr,g,xi,ds,mp):
    que=10
    bot = mp**2 + que
    return (mp**2 / bot)*Fvector2(P_input,m1,m2,mr,g,xi,ds)

#This function doesnt work properly, nonzero imag
def Fnos(P_i,P_f,m1,m2,mr,g,xi,ds,mp,B,C,que):
    # que=0
    si = dotprod(P_i,P_i)
    sf = dotprod(P_f,P_f)
    front = (1/2 + B*(si-sf))*Fvectorsi(P_i,m1,m2,mr,g,xi,que)
    back = (1/2 + C*(sf-si))*Fvectorsf(P_f,m1,m2,mr,g,xi,que)
    return back+front

#Use this function and edit the q2 in F_
def Fnos0(P_i,P_f,m1,m2,mr,g,xi,ds,mp,B,C):
    # que=0
    si = dotprod(P_i,P_i)
    sf = dotprod(P_f,P_f)
    front = (1/2 + B*(si-sf))*F_0(P_i,m1,m2,mr,g,xi,ds,mp)
    back = (1/2 + C*(sf-si))*F_0(P_f,m1,m2,mr,g,xi,ds,mp)
    return back+front

#these following function correspond to the remaining idea I have for fixed noise on F.
#FIRST we need rho function and K inverse

def rho(s,m1,m2,xi):
    top = xi*qStar(m1,m2,s)
    bottom = 8*math.pi*np.sqrt(s)
    if bottom == 0+0j:
        return np.inf
    # if qStar(m1,m2,s) < 0:
    #     return -top/bottom
    # if qStar(m1,m2,s) > 0:
    return top/bottom

def Kinv(s,m1,m2,xi,mr,g):
    return rho(s,m1,m2,xi)*(1/tanBW(m1,m2,g,s,mr))

#here is the culmination of all the testing bs

def cleansmoothF(s,m1,m2,xi,mr,g):
    one = -(3/2)*xi*csqrt(s)/(g**2*mr**2)
    # one = 2*csqrt(s)*dbydx(Kinv,s,ds,(m1,m2,xi,mr,g))
    two = -(xi*1j)/(32*math.pi*qStar(m1,m2,s))
    three = 1j*(qStar(m1,m2,s)*xi)/(8*math.pi*s)
    four = -(1j/(32*math.pi*s*qStar(m1,m2,s)))*(((m1**2-m2**2)**2)/s)
    five = (1j/(16*math.pi*s*qStar(m1,m2,s)))*(m1**2+m2**2)
    return (one+two+three+four+five)

#we need to define a step func to use in the future, could be moved to the top of this code
def stepfunction(x):
    if x < 0:
        return 0
    return 1

def kallen(x1,x2,x3):
    one = pow(x1,2)
    two = pow(x2,2)
    three = pow(x3,2)
    four = 2*x1*x2
    five = 2*x1*x3
    six = 2*x2*x3
    return one+two+three-four-five-six

def Aif(m1,m2,Pi,Pf):
    si = dotprod(Pi,Pi)
    sf = dotprod(Pf,Pf)
    diff = np.subtract(Pf,Pi)
    q2 = dotprod(diff,diff)

    Ei = (1/(2*math.sqrt(sf)))*(sf+si-q2)
    Ef = (1/(2*math.sqrt(si)))*(sf+si-q2)
    wi = (1/(2*math.sqrt(sf)))*(sf+pow(m1,2)-pow(m2,2))
    wf = (1/(2*math.sqrt(si)))*(si+pow(m1,2)-pow(m2,2))
    absPi = (1/(2*math.sqrt(sf)))*cmath.sqrt(kallen(sf,si,q2))
    absPf = (1/(2*math.sqrt(si)))*cmath.sqrt(kallen(sf,si,q2))
    qf = (1/(2*math.sqrt(sf)))*cmath.sqrt(kallen(sf,pow(m1,2),pow(m2,2)))
    qi = (1/(2*math.sqrt(si)))*cmath.sqrt(kallen(si,pow(m1,2),pow(m2,2)))

    numer = si - (2*Ei*wi) + pow(m1,2) - pow(m2,2)
    denom = 2*absPi*qf

    return -1*(numer/denom)

def Afi(m1,m2,Pi,Pf):
    si = dotprod(Pi,Pi)
    sf = dotprod(Pf,Pf)
    diff = np.subtract(Pf,Pi)
    q2 = dotprod(diff,diff)

    Ei = (1/(2*math.sqrt(sf)))*(sf+si-q2)
    Ef = (1/(2*math.sqrt(si)))*(sf+si-q2)
    wi = (1/(2*math.sqrt(sf)))*(sf+pow(m1,2)-pow(m2,2))
    wf = (1/(2*math.sqrt(si)))*(si+pow(m1,2)-pow(m2,2))
    absPi = (1/(2*math.sqrt(sf)))*cmath.sqrt(kallen(sf,si,q2))
    absPf = (1/(2*math.sqrt(si)))*cmath.sqrt(kallen(sf,si,q2))
    qf = (1/(2*math.sqrt(sf)))*cmath.sqrt(kallen(sf,pow(m1,2),pow(m2,2)))
    qi = (1/(2*math.sqrt(si)))*cmath.sqrt(kallen(si,pow(m1,2),pow(m2,2)))

    numer = sf - (2*Ef*wf) + pow(m1,2) - pow(m2,2)
    denom = 2*absPf*qi

    return -1*(numer/denom)

def G0cut(m1,m2,Pi,Pf):
    si = dotprod(Pi,Pi)
    sf = dotprod(Pf,Pf)
    diff = np.subtract(Pf,Pi)
    q2 = dotprod(diff,diff)
    quant = (dotprod(Pf,Pi))**2 - (si*sf)
    coeff = (1/(32*math.pi*cmath.sqrt(quant)))

    lead = math.log(np.abs((Aif(m1,m2,Pi,Pf) +1)/(Aif(m1,m2,Pi,Pf)-1)))+(1j*math.pi*(stepfunction(Aif(m1,m2,Pi,Pf)+1)-stepfunction(Aif(m1,m2,Pi,Pf)-1)))
    follow = math.log(np.abs((Afi(m1,m2,Pi,Pf) +1)/(Afi(m1,m2,Pi,Pf)-1)))+(1j*math.pi*(stepfunction(Afi(m1,m2,Pi,Pf)+1)-stepfunction(Afi(m1,m2,Pi,Pf)-1)))

    return coeff*(lead+follow)

def G0cutrotated(m1,m2,Pi,Pf):
    si = dotprod(Pi,Pi)
    sf = dotprod(Pf,Pf)
    diff = np.subtract(Pf,Pi)
    q2 = dotprod(diff,diff)
    quant = (dotprod(Pf,Pi))**2 - (si*sf)
    coeff = (1j/(32*math.pi*cmath.sqrt(quant)))

    lead = np.log(-1j*(Aif(m1,m2,Pi,Pf)+1))- np.log(1j*(Aif(m1,m2,Pi,Pf)-1))
    follow = np.log(-1j*(Afi(m1,m2,Pi,Pf)+1))- np.log(1j*(Afi(m1,m2,Pi,Pf)-1))

    return coeff*(lead+follow)

#here will begin my work on the contour plots.
#the first step is to redine M in terms of rho
#we will use rho and Kinv from above

def Kmatrix(s,m1,m2,xi,mr,g):
    return tanBW(m1,m2,g,s,mr)/rho(s,m1,m2,xi)

def Kmatrixnosing(s,m1,m2,xi,mr,g):
    top = 4*pow(g,2)*pow(mr,2)
    bottom = 3*(pow(mr,2)-s)
    return top/bottom

def MrhoKnoninv(s,m1,m2,xi,mr,g):
    lead = Kmatrix(s,m1,m2,xi,mr,g)
    follow = 1/(1 - (1j*rho(s,m1,m2,xi) * Kmatrix(s,m1,m2,xi,mr,g)))
    return lead * follow
