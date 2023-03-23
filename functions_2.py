import math
import cmath
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


def csqrt(z):
	arg = np.angle(z)
	norm = np.absolute(z)
	newArg = np.mod(arg + 2*math.pi, 2*math.pi)
	return np.sqrt(norm)*np.exp(1j*newArg/2)

def csqrt_old(z):
	return 1j*np.sqrt(-z)

def dot(v1,v2):
	termOne = v1[0]*v2[0]
	termTwo = v1[1]*v2[1]
	termThree = v1[2]*v2[2]
	termFour = v1[3]*v2[3]
	return termOne - termTwo - termThree - termFour

def derivative(func, x, dx, args=()):
	numer = func(x+dx,*args) - func(x-dx,*args)
	denum = 2*dx
	return numer/denum

def qStar(s,m1,m2):
	if s == 0+0j:
		return np.inf
	termOne = 2*(pow(m1,2) + pow(m2,2))
	termTwo = pow(pow(m2,2) - pow(m1,2),2)/s
	return 0.5*csqrt(s - termOne + termTwo)

def gammaS(s,m1,m2,mr,g):
	if s == 0+0j:
		return np.inf
	amp = pow(g,2)/( 6.0 * math.pi )
	return amp * pow(mr,2) * (qStar(s,m1,m2)) / s

def gammaP(s,m1,m2,g):
	if s == 0+0j:
		return np.inf
	amp = pow(g,2)/( 6.0 * math.pi )
	return amp * pow(qStar(s,m1,m2),3)/s

def tanBW_func(s,m1,m2,mr,g,l):
	denum = pow(mr,2)-s
	if l == 0:
		Gamma = gammaS(s,m1,m2,mr,g)
	elif l == 1:
		Gamma = gammaP(s,m1,m2,g)
	return csqrt(s)*Gamma / denum

def rho(s,m1,m2,xi):
	numer = xi*qStar(s,m1,m2)
	denum = 8*math.pi*cmath.sqrt(s)
	if denum == 0+0j:
		return np.inf
	return numer/denum

def f(mr, Q2):
	return pow(mr,2)/(pow(mr,2)+Q2)

def K(s,m1,m2,xi,mr,g,l):
	numer = 8*math.pi*csqrt(s)*tanBW_func(s,m1,m2,mr,g,l)
	denum = xi*qStar(s,m1,m2)
	return numer/denum

def M(s,m1,m2,xi,mr,g,l=0):
	if s == 0+0j:
		return np.inf
	# print("s: " + str(s))
	kInverse = 1/K(s,m1,m2,xi,mr,g,l)
	denum = kInverse - (1j)*rho(s,m1,m2,xi)
	# print("denum: " + str(denum))
	return 1/denum

def Mere(s,m1,m2,xi,a,r):
	qCotDelta = -1.0 / a + 0.5 * r * qStar(s,m1,m2)**2
	K = ( 8.0 * np.pi * np.sqrt(s) / xi ) / qCotDelta
	p = rho(s,m1,m2,xi)
	return K / ( 1.0 - 1j * p * K )

def MII(s,m1,m2,xi,mr,g,l=0):
	# print("s: " + str(s))
	kInverse = 1/K(s,m1,m2,xi,mr,g,l)
	denum = kInverse + (1j)*rho(s,m1,m2,xi)
	# print("denum: " + str(denum))
	return 1/denum

def H(s,m1,m2,xi,mr,g,Q2):
	ftmp = f(mr,Q2)
	F = ftmp/3
	return F*M(s,m1,m2,xi,mr,g)

def F(mr,Q2,g,scale):
	return (scale/pow(g,2))*f(mr,Q2)

def A(x,m1,m2,Q2,si,sf):
	numer = pow(m2,2) - pow(m1,2) - x*(Q2+sf+si)
	return 1 + numer/si

def B(x,m1,m2,si,sf):
	numer = pow(m2,2) - x*(pow(m2,2) - pow(m1,2)) - x*(1-x)*sf
	return -4*(numer/si)

def yplus(x,m1,m2,Q2,si,sf,epsilon):
	return 0.5*(A(x,m1,m2,Q2,si,sf) + cmath.sqrt((A(x,m1,m2,Q2,si,sf)**2) + B(x,m1,m2,si,sf)) + complex(0,epsilon))

def yminus(x,m1,m2,Q2,si,sf,epsilon):
	return 0.5*(A(x,m1,m2,Q2,si,sf) - cmath.sqrt((A(x,m1,m2,Q2,si,sf)**2) + B(x,m1,m2,si,sf)) + complex(0,epsilon))

def Lp(x,y,epsilon):
	logpart = math.log(abs((1-x-y)/y))
	arctanpart1 = 0
	arctanpart2 = 0
	a = 1-x-y.real

	if y.imag == 0:
		if a >= 0:
			arctanpart1 =  cmath.pi/2
		else:
			arctanpart1 = -1*cmath.pi/2

		if y.real >= 0:
			arctanpart2 =  cmath.pi/2
		else:
			arctanpart2 = -1*cmath.pi/2
	else:
		arctanpart1 = math.atan(a/(y.imag+epsilon))
		arctanpart2 = math.atan(y.real/(y.imag+epsilon))

	return complex(logpart,(arctanpart1 + arctanpart2))

def Lm(x,y,epsilon):
	logpart = math.log(abs((1-x-y)/y))
	arctanpart1 = 0
	arctanpart2 = 0
	a = 1-x-y.real

	if y.imag == 0 and epsilon == 0:
		if a >= 0:
			arctanpart1 =  -1*cmath.pi/2
		else:
			arctanpart1 = cmath.pi/2

		if y.real >= 0:
			arctanpart2 =  -1*cmath.pi/2
		else:
			arctanpart2 = cmath.pi/2
	else:
		arctanpart1 = math.atan(a/(y.imag-epsilon))
		arctanpart2 = math.atan(y.real/(y.imag-epsilon))

	return complex(logpart,(arctanpart1 + arctanpart2))

def realIntegrandG0(x,m1,m2,Q2,si,sf,epsilon):
	coefficient = 1/(pow(4*math.pi,2)*si)
	yp = yplus(x,m1,m2,Q2,si,sf,epsilon)
	ym = yminus(x,m1,m2,Q2,si,sf,epsilon)
	diffTop = Lp(x,yp,epsilon) - Lm(x,ym,epsilon)
	diffBottom = yp-ym
	if diffBottom == 0:
		return math.inf

	return (coefficient*diffTop/diffBottom).real

def imagIntegrandG0(x,m1,m2,Q2,si,sf,epsilon):
	coefficient = 1/(pow(4*math.pi,2)*si)
	yp = yplus(x,m1,m2,Q2,si,sf,epsilon)
	ym = yminus(x,m1,m2,Q2,si,sf,epsilon)
	diffTop = Lp(x,yp,epsilon) - Lm(x,ym,epsilon)
	diffBottom = yp-ym
	if diffBottom == 0:
		return math.inf

	return (coefficient*diffTop/diffBottom).imag

def realIntegrandG11(x,m1,m2,Q2,si,sf,epsilon):
	F1 = realIntegrandG0(x,m1,m2,Q2,si,sf,epsilon)
	return x*F1

def imagIntegrandG11(x,m1,m2,Q2,si,sf,epsilon):
	F1 = imagIntegrandG0(x,m1,m2,Q2,si,sf,epsilon)
	return x*F1

def realIntegrandG21(x,m1,m2,Q2,si,sf,epsilon):
	F1 = realIntegrandG0(x,m1,m2,Q2,si,sf,epsilon)
	return pow(x,2)*F1

def imagIntegrandG21(x,m1,m2,Q2,si,sf,epsilon):
	F1 = imagIntegrandG0(x,m1,m2,Q2,si,sf,epsilon)
	return pow(x,2)*F1

def realIntegrandG12(x,m1,m2,Q2,si,sf,epsilon):
	coefficient = 1/(pow(4*math.pi,2)*si)
	yp = yplus(x,m1,m2,Q2,si,sf,epsilon)
	ym = yminus(x,m1,m2,Q2,si,sf,epsilon)
	diffTop = (yp*Lp(x,yp,epsilon)) - (ym*Lm(x,ym,epsilon))
	diffBottom = yp-ym
	if diffBottom == 0:
		return math.inf

	return (coefficient*diffTop/diffBottom).real

def imagIntegrandG12(x,m1,m2,Q2,si,sf,epsilon):
	coefficient = 1/(pow(4*math.pi,2)*si)
	yp = yplus(x,m1,m2,Q2,si,sf,epsilon)
	ym = yminus(x,m1,m2,Q2,si,sf,epsilon)
	diffTop = (yp*Lp(x,yp,epsilon)) - (ym*Lm(x,ym,epsilon))
	diffBottom = yp-ym
	if diffBottom == 0:
		return math.inf

	return (coefficient*diffTop/diffBottom).imag

def quadraticSolution(a,b,c):
	sqrtTerm = math.sqrt(pow(b,2)-4*a*c)
	point1 = (-b-sqrtTerm)/(2*a)
	point2 = (-b+sqrtTerm)/(2*a)
	return point1, point2

def criticalPoints(m1,m2,Q2,si,sf):
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

def G0(si,Q2,sf,m1,m2,epsilon):
	# if sfi != 0:
	# 	sf = complex(sfr,sfi)
	# else:
	# 	sf = sfr
	critical = criticalPoints(m1,m2,Q2.real,si.real,sf.real)
	real = integrate.quad(realIntegrandG0,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	imag = integrate.quad(imagIntegrandG0,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	return real, imag

def G0II(si,Q2,sf,m1,m2,epsilon):
	critical = criticalPoints(m1,m2,Q2,si.real,sf.real)
	real = integrate.quad(realIntegrandG0,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	imag = integrate.quad(imagIntegrandG0,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	G = complex(real,imag)
	Gii = G-2j*G.imag 
	return Gii.real, Gii.imag

def G11(si,Q2,sf,m1,m2,epsilon):
	critical = criticalPoints(m1,m2,Q2.real,si.real,sf.real)
	real = integrate.quad(realIntegrandG11,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	imag = integrate.quad(imagIntegrandG11,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	return real, imag

def G12(si,Q2,sf,m1,m2,epsilon):
	critical = criticalPoints(m1,m2,Q2.real,si.real,sf.real)
	real = integrate.quad(realIntegrandG12,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	imag = integrate.quad(imagIntegrandG12,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	return real, imag

def G21(si,Q2)

def Gvector(Pi,Pf,m1,m2,epsilon):
	si = dot(Pi,Pi)
	sf = dot(Pf,Pf)
	q = np.subtract(Pf,Pi)
	Q2 = -1*dot(q,q)
	g0real, g0imag = G0(si,Q2,sf,m1,m2,epsilon)
	g11real, g11imag = G11(si,Q2,sf,m1,m2,epsilon)
	g12real, g12imag = G12(si,Q2,sf,m1,m2,epsilon)
	g0 = complex(g0real, g0imag)
	g11 = complex(g11real, g11imag)
	g12 = complex(g12real, g12imag)
	firstTerm = Pi*(g0-2*g12)
	secondTerm = Pf*(g0-2*g11)
	# print("m1: " + str(m1) + ", m2: " + str(m2))
	# print("Pi: " + str(Pi) + ", Pf: " + str(Pf))
	# print("g0: " + str(g0))
	# print("g11: " + str(g11))
	# print("g12: " + str(g12))
	# print("First: " + str(firstTerm[0]))
	# print("Second: " + str(secondTerm[0]))
	gvec = firstTerm + secondTerm
	return gvec

def Gvector_II(Pi,Pf,m1,m2,epsilon):
	gvec = Gvector(Pi,Pf,m1,m2,epsilon)
	gvec_II = gvec - 2*(1j)*gvec.imag
	return gvec_II


def Ia_nu(Pi, Pf, m1, m2, epsilon):
	si = dot(Pi, Pi)
	sf = dot(Pf, Pf)
	q = np.subtract(Pf, Pi)
	Q2 = -1*dot(q, q)
	g11real, g11imag = G11(si, Q2, sf, m1, m2, epsilon)
	g12real, g12imag = G12(si, Q2, sf, m1, m2, epsilon)
	g11 = complex(g11real, g11imag)
	g12 = complex(g12real, g12imag)
	firstTerm = Pi*g12
	secondTerm = Pf*g11
	# print("m1: " + str(m1) + ", m2: " + str(m2))
	# print("Pi: " + str(Pi) + ", Pf: " + str(Pf))
	# print("g11: " + str(g11))
	# print("g12: " + str(g12))
	# print("First: " + str(firstTerm[0]))
	# print("Second: " + str(secondTerm[0]))
	gvec = firstTerm + secondTerm
	return gvec.real, gvec.imag

def dMds(s,m1,m2,mr,g,xi):
	q = qStar(s,m1,m2)
	m1Sq = pow(m1,2)
	m2Sq = pow(m2,2)
	massSqDiff = pow(m1Sq - m2Sq,2)
	massSqSum = m1Sq + m2Sq
	gmrSq = pow(g,2)*pow(mr,2)

	numerCoeff = 8*gmrSq*math.pi
	numerTermOne = gmrSq*(massSqDiff - massSqSum*s)*1j
	numerTermTwo = -24*math.pi*pow(s,5/2)*q

	denumCoeff = xi*pow(s,3/2)*q
	denumTermOne = pow(complex(gmrSq*2*q, 12*math.pi*(pow(mr,2)-s)*math.sqrt(s)),2)

	numer = numerCoeff*(numerTermOne + numerTermTwo)
	denum = denumCoeff*denumTermOne

	return numer/denum

def Wvector_old(P,m1,m2,mr,g,xi):
	s = dot(P,P)
	Wvec = 2*P*dMds(s,m1,m2,mr,g,xi)
	return Wvec.real, Wvec.imag

def Wvector(P,m1,m2,mr,g,xi,ds):
	s = dot(P,P)
	dmds = derivative(M,s,ds,(m1,m2,xi,mr,g))
	Wvec = 4*P*dmds
	return Wvec.real, Wvec.imag
	
def Fvector(P,m1,m2,mr,g,xi,ds,epsilon):
	s = dot(P,P)
	if s == 0+0j:
		return np.array([0,0,0,0])
	# q = qStar(s,m1,m2)
	m = M(s,m1,m2,xi,mr,g)
	WvecReal, WvecImag = Wvector(P,m1,m2,mr,g,xi,ds)
	Wvec = np.array([complex(WvecReal[i], WvecImag[i]) for i in range(len(WvecReal))])
	termOne = (1/m)*Wvec*(1/m)
	Gvec = Gvector(P,P,m1,m2,epsilon)
	# print("Before: " + str(Gvec))
	# gret = []
	# for i in Gvec:
	# 	gret.append(complex(0,i.imag))

	# g = np.array(gret)
	# print("After: " + str(gret))
	# GvecReal, GvecImag = Gvector(P,P,m1,m2,epsilon)
	# Gvec = np.array([complex(GvecReal[i], GvecImag[i]) for i in range(len(GvecReal))])
	lf = f(mr,0)
	termTwo = lf*Gvec
	# termTwo = lf*g
	Fvec = termOne - termTwo
	return Fvec
	
def Fvector_2(P,m1,m2,mr,g,xi,ds,epsilon):
	s = dot(P,P)
	if s == 0+0j:
		return np.array([0,0,0,0])
	# q = qStar(s,m1,m2)
	m = M(s,m1,m2,xi,mr,g)
	WvecReal, WvecImag = Wvector(P,m1,m2,mr,g,xi,ds)
	Wvec = np.array([complex(WvecReal[i], WvecImag[i]) for i in range(len(WvecReal))])
	termOne = (1/m)*Wvec*(1/m)
	Gvec = Gvector(P,P,m1,m2,epsilon)
	lf = f(mr,0)
	termTwo = lf*Gvec
	return termOne, termTwo

def Fvector_3(P,m,mr,g):
	coeffOne = 3/(2*pow(g,2)*pow(mr,2))
	termOne = coeffOne*P
	Gvec = Gvector(P,P,m,m,0)

	thresh_s = pow(2*m,2)
	s = dot(P,P)
	if s > thresh_s:
		return termOne - Gvec.real
	coeffTwo = thresh_s/(32*math.pi*pow(s,3/2)*abs(qStar(s,m,m)))
	termTwo = coeffTwo*P
	return termOne + termTwo - Gvec.real


def Avector(B,Pi,Pf,m1,m2,mr,g,xi,ds,epsilon):
	si = dot(Pi,Pi)
	sf = dot(Pf,Pf)
	Fi = Fvector(Pi,m1,m2,mr,g,xi,ds,epsilon)
	Ff = Fvector(Pf,m1,m2,mr,g,xi,ds,epsilon)
	# print("Fi: " + str(Fi))
	# print("Ff: " + str(Ff))
	coeff_i = 0.5 + B*(si-sf)
	coeff_f = 0.5 + B*(sf-si)
	numer1 = coeff_i*Fi
	numer2 = coeff_f*Ff
	numer = numer1 + numer2

	q = np.subtract(Pf, Pi)
	Q2 = -1*dot(q, q)
	denum = 1 + (Q2/pow(mr,2))
	return numer/denum

def Avector_2(B,Pi,Pf,m,mr,g):
	si = dot(Pi,Pi)
	sf = dot(Pf,Pf)
	Fi = Fvector_3(Pi,m,mr,g)
	Ff = Fvector_3(Pf,m,mr,g)
	# print("Fi: " + str(Fi))
	# print("Ff: " + str(Ff))
	coeff_i = 0.5 + B*(si-sf)
	coeff_f = 0.5 + B*(sf-si)
	numer1 = coeff_i*Fi
	numer2 = coeff_f*Ff
	numer = numer1 + numer2

	q = np.subtract(Pf, Pi)
	Q2 = -1*dot(q, q)
	denum = 1 + (Q2/pow(mr,2))
	return numer/denum

def triangle(a,b,c):
	termOne = pow(a,2)
	termTwo = pow(b,2)
	termThree = pow(c,2)
	termFour = 2*a*b
	termFive = 2*a*c
	termSix = 2*b*c
	return termOne + termTwo + termThree - termFour - termFive - termSix

def stepFunc(x):
	if x < 0:
		return 0
	return 1

def afi(Pi,Pf,m1,m2):
	si = dot(Pi, Pi)
	sf = dot(Pf, Pf)
	q = np.subtract(Pf, Pi)
	q2 = dot(q, q)
	m1sq = pow(m1, 2)
	m2sq = pow(m2, 2)

	coeffI = 1/(2*math.sqrt(si))
	qi = coeffI*cmath.sqrt(triangle(si, m1sq, m2sq))
	omega1i = coeffI*(si + m1sq - m2sq)
	ef = coeffI*(sf + si - q2)
	pfvec = coeffI*cmath.sqrt(triangle(sf, si, q2))

	Afi_numer = sf - 2*ef*omega1i + m1sq - m2sq
	Afi_denum = 2*pfvec*qi
	Afi = -1*(Afi_numer/Afi_denum)

	return Afi

def aif(Pi,Pf,m1,m2):
	si = dot(Pi, Pi)
	sf = dot(Pf, Pf)
	q = np.subtract(Pf, Pi)
	q2 = dot(q, q)
	m1sq = pow(m1, 2)
	m2sq = pow(m2, 2)

	coeffF = 1/(2*math.sqrt(sf))
	qf = coeffF*cmath.sqrt(triangle(sf, m1sq, m2sq))
	omega1f = coeffF*(sf + m1sq - m2sq)
	ei = coeffF*(sf + si - q2)
	pivec = coeffF*cmath.sqrt(triangle(sf, si, q2))

	Aif_numer = si - 2*ei*omega1f + m1sq - m2sq
	Aif_denum = 2*pivec*qf
	Aif = -1*(Aif_numer/Aif_denum)

	return Aif


def G0cutkosky(Pi,Pf,m1,m2):
	si = dot(Pi,Pi)
	sf = dot(Pf,Pf)
	q = np.subtract(Pf,Pi)
	q2 = dot(q,q)
	m1sq = pow(m1,2)
	m2sq = pow(m2,2)

	# When Pf=0
	coeffF = 1/(2*math.sqrt(sf))
	qf = coeffF*cmath.sqrt(triangle(sf,m1sq,m2sq))
	omega1f = coeffF*(sf + m1sq - m2sq)
	ei = coeffF*(sf + si - q2)
	pivec = coeffF*cmath.sqrt(triangle(sf,si,q2))

	Aif_numer = si - 2*ei*omega1f + m1sq - m2sq
	Aif_denum = 2*pivec*qf
	Aif = -1*(Aif_numer/Aif_denum)

	# When Pi=0
	coeffI = 1/(2*math.sqrt(si))
	qi = coeffI*cmath.sqrt(triangle(si, m1sq, m2sq))
	omega1i = coeffI*(si + m1sq - m2sq)
	ef = coeffI*(sf + si - q2)
	pfvec = coeffI*cmath.sqrt(triangle(sf, si, q2))

	Afi_numer = sf - 2*ef*omega1i + m1sq - m2sq
	Afi_denum = 2*pfvec*qi
	Afi = -1*(Afi_numer/Afi_denum)

	# Putting it all together
	# ep = csqrt(pow(dot(Pf, Pi), 2)-si*sf)
	ep = cmath.sqrt(pow(dot(Pf, Pi), 2)-si*sf)
	if ep == 0:
		print("ep is 0")
		coeff = 1/(32*math.pi*0.000000001)
	else:
		coeff = 1/(32*math.pi*ep)

	# coeff = 1/(32*math.pi*ep + 1j*0.000000001)

	# tmp1 = stepFunc(Aif+1)
	# tmp2 = stepFunc(Aif-1)
	# tmp3 = stepFunc(Afi+1)
	# tmp4 = stepFunc(Afi+1)

	termOne = math.log(abs((Aif+1)/(Aif-1))) + 1j*math.pi*(stepFunc(Aif+1) - stepFunc(Aif-1))
	termTwo = math.log(abs((Afi+1)/(Afi-1))) + 1j*math.pi*(stepFunc(Afi+1) - stepFunc(Afi-1))
	G0 = coeff*(termOne + termTwo)

	return G0.real, G0.imag
	

def G0cutkoskyRot(Pi, Pf, m1, m2):
	si = dot(Pi, Pi)
	sf = dot(Pf, Pf)
	q = np.subtract(Pf, Pi)
	q2 = dot(q, q)
	m1sq = pow(m1, 2)
	m2sq = pow(m2, 2)

	# When Pf=0
	coeffF = 1/(2*math.sqrt(sf))
	qf = coeffF*cmath.sqrt(triangle(sf, m1sq, m2sq))
	omega1f = coeffF*(sf + m1sq - m2sq)
	ei = coeffF*(sf + si - q2)
	pivec = coeffF*cmath.sqrt(triangle(sf, si, q2))

	Aif_numer = si - 2*ei*omega1f + m1sq - m2sq
	Aif_denum = 2*pivec*qf
	Aif = -1*(Aif_numer/Aif_denum)

	# When Pi=0
	coeffI = 1/(2*math.sqrt(si))
	qi = coeffI*cmath.sqrt(triangle(si, m1sq, m2sq))
	omega1i = coeffI*(si + m1sq - m2sq)
	ef = coeffI*(sf + si - q2)
	pfvec = coeffI*cmath.sqrt(triangle(sf, si, q2))

	Afi_numer = sf - 2*ef*omega1i + m1sq - m2sq
	Afi_denum = 2*pfvec*qi
	Afi = -1*(Afi_numer/Afi_denum)

	# Putting it all together
	ep = cmath.sqrt(pow(dot(Pf, Pi), 2)-si*sf)
	# if ep == 0:
	# 	print("ep is 0")
	# 	coeff = 1/(32*math.pi*0.000000001)
	# else:
	# 	coeff = 1/(32*math.pi*ep)

	coeff = 1j/(32*math.pi*ep + 1j*0.000000001)

	termOne = np.log(-1j*(Aif+1)) - np.log(1j*(Aif-1)) #+ 1j*math.pi * (stepFunc(Aif+1) - stepFunc(Aif-1))
	termTwo = np.log(-1j*(Afi+1)) - np.log(1j*(Afi-1)) #+ 1j*math.pi*(stepFunc(Afi+1) - stepFunc(Afi-1))
	G0 = coeff*(termOne + termTwo)

	return G0.real, G0.imag

def GI(s,m1,m2):
	coeff = 1j*(1/(32*math.pi*cmath.sqrt(s)*qStar(s,m1,m2)))
	factorOne = (s-pow(m2,2)+pow(m1,2))/s
	return coeff*factorOne


def Wdf_vector(Pi,Pf,Q2,m1,m2,mr,xi,g,B,sheet):
	si = dot(Pi,Pi)
	sf = dot(Pf,Pf)
	M1 = M(si,m1,m2,xi,mr,g) if sheet == 1 else MII(si,m1,m2,xi,mr,g)
	A = Avector_2(B,Pi,Pf,m1,mr,g)
	fs = f(mr,Q2)
	G = Gvector(Pi,Pf,m1,m2,0) if sheet == 1 else Gvector_II(Pi,Pf,m1,m2,0)
	M2 = M(sf,m1,m2,xi,mr,g) if sheet == 1 else MII(sf,m1,m2,xi,mr,g)
	return M1*(A+fs*G)*M2













