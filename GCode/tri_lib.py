#! /usr/bin/env python

import cmath
import math
import numpy as np
import scipy.integrate as integrate

# General Helper Functions
#region

gmn = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])

def dot(v1,v2):
	termOne = v1[0]*v2[0]
	termTwo = v1[1]*v2[1]
	termThree = v1[2]*v2[2]
	termFour = v1[3]*v2[3]
	return termOne - termTwo - termThree - termFour

def tri(a,b,c):
	return pow(a,2) + pow(b,2) + pow(c,2) -2*(a*b + b*c + c*a)

def csqrt(z):
	arg = np.angle(z)
	norm = np.absolute(z)
	newArg = np.mod(arg + 2*math.pi, 2*math.pi)
	return np.sqrt(norm)*np.exp(1j*newArg/2)

def qStar(s,m1,m2):
	if s == 0+0j:
		return np.inf
	termOne = 2*(pow(m1,2) + pow(m2,2))
	termTwo = pow(pow(m2,2) - pow(m1,2),2)/s
	return 0.5*csqrt(s - termOne + termTwo)

def isZeroVec(v):
	if v[0] == 0 and v[1]==0 and v[2]==0:
		return True
	return False

#endregion

# Lorentz Boost
#region

def unit_vector( A ):
    if( all( A==0 ) ):
        return A
    else:
        return A / np.sqrt( np.dot(A,A) )

def calcBeta(P):
    tmp = np.array([P[1],P[2],P[3]])
    return tmp/P[0]

def lorentzBoost( P, beta ):
    A0 = P[0]
    A = np.array([P[1],P[2],P[3]])
    gamma = 1.0 / np.sqrt( 1.0 - np.dot(beta,beta) )
    Apar  = A.dot( unit_vector(beta) )*unit_vector(beta)
    Aper  = A - Apar
    A0_prime = gamma * ( A0 - A.dot( beta ) )
    A_prime  = gamma * ( Apar - A0*beta  ) + Aper
    return np.array([A0_prime, round(A_prime[0],6), round(A_prime[1],6), round(A_prime[2],6)])

# Boost Matrix Helper
def BMH(beta1, beta2, betaSq):
	gamma = 1/np.sqrt(1.0-betaSq)
	numer = (gamma-1)*beta1*beta2
	return numer/betaSq

def BoostMatrixB(beta):
	if isZeroVec(beta):
		return np.identity(4)
	betaSq = np.dot(beta,beta)
	gamma = 1/np.sqrt(1.0-betaSq)
	dim = np.arange(1,4)
	Lambda = np.zeros((len(dim)+1,len(dim)+1))
	Lambda[0][0] = gamma
	for i in dim:
		Lambda[0][i]=Lambda[i][0]=-gamma*beta[i-1]
	for i in dim:
		for j in dim:
			if j < i:
				continue
			tmp = BMH(beta[i-1],beta[j-1],betaSq)
			if i == j:
				Lambda[i][j] = 1 + tmp
			else:
				Lambda[i][j] = Lambda[j][i] = tmp
	return Lambda

def BoostMatrixP(P):
	return BoostMatrixB(calcBeta(P))

#endregion

# G sub-functions
#region

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

#endregion

# Integrations
#region

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

def integrandG0(x,m1,m2,Q2,si,sf,epsilon):
	coefficient = 1/(pow(4*math.pi,2)*si)
	yp = yplus(x,m1,m2,Q2,si,sf,epsilon)
	ym = yminus(x,m1,m2,Q2,si,sf,epsilon)
	diffTop = Lp(x,yp,epsilon) - Lm(x,ym,epsilon)
	diffBottom = yp-ym
	if diffBottom == 0:
		return math.inf

	return (coefficient*diffTop/diffBottom)


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

def realIntegrandG23(x,m1,m2,Q2,si,sf,epsilon):
	return x*realIntegrandG12(x,m1,m2,Q2,si,sf,epsilon)

def imagIntegrandG23(x,m1,m2,Q2,si,sf,epsilon):
	return x*imagIntegrandG12(x,m1,m2,Q2,si,sf,epsilon)

def realIntegrandG24(x,m1,m2,Q2,si,sf,epsilon):
	coefficient = -1/(8*pow(math.pi,2))
	yp = yplus(x,m1,m2,Q2,si,sf,epsilon)
	ym = yminus(x,m1,m2,Q2,si,sf,epsilon)
	log1 = cmath.log(1-x-ym)
	log2 = cmath.log(1-x-yp)
	log3 = cmath.log(-ym)
	log4 = cmath.log(-yp)
	return (coefficient*((1-x-ym)*log1 + (1-x-yp)*log2 + ym*log3 + yp*log4)).real

def imagIntegrandG24(x,m1,m2,Q2,si,sf,epsilon):
	coefficient = -1/(8*pow(math.pi,2))
	yp = yplus(x,m1,m2,Q2,si,sf,epsilon)
	ym = yminus(x,m1,m2,Q2,si,sf,epsilon)
	log1 = cmath.log(1-x-ym)
	log2 = cmath.log(1-x-yp)
	log3 = cmath.log(-ym)
	log4 = cmath.log(-yp)
	return (coefficient*((1-x-ym)*log1 + (1-x-yp)*log2 + ym*log3 + yp*log4)).imag

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
	# real = integrate.quad(realIntegrandG0,0,1,full_output=1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	# imag = integrate.quad(imagIntegrandG0,0,1,full_output=1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	real = integrate.quad(realIntegrandG0,0,1,full_output=1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)
	imag = integrate.quad(imagIntegrandG0,0,1,full_output=1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)
	# full = integrate.quad(integrandG0,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	return real, imag

def G0II(si,Q2,sf,m1,m2,epsilon):
	critical = criticalPoints(m1,m2,Q2,si.real,sf.real)
	real = integrate.quad(realIntegrandG0,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	imag = integrate.quad(imagIntegrandG0,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	G = complex(real,imag)
	Gii = G-2j*G.imag 
	return Gii.real, Gii.imag

def I23(si,Q2,sf,m1,m2,epsilon):
	critical = criticalPoints(m1,m2,Q2,si.real,sf.real)
	real = integrate.quad(realIntegrandG23,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	imag = integrate.quad(imagIntegrandG23,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	return complex(real,imag)

def I24(si,Q2,sf,m1,m2,epsilon):
	critical = criticalPoints(m1,m2,Q2,si.real,sf.real)
	real = integrate.quad(realIntegrandG24,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	imag = integrate.quad(imagIntegrandG24,0,1,args=(m1,m2,Q2,si,sf,epsilon),points=critical)[0]
	return complex(real,imag)
#endregion

# P-wave Scalar G
#region

def MomentumMat(Pif,Pfi):
	coeff1 = 1/2
	coeff2 = 1/math.sqrt(2)
	gn1n1 = coeff1*((Pif[1]*Pfi[1] + Pif[2]*Pfi[2]) -1j*(Pif[1]*Pfi[2] - Pif[2]*Pfi[1]))
	gn10 = coeff2*(Pif[1]*Pfi[3] +1j*(Pif[2]*Pfi[3]))
	gn11 = -1*coeff1*((Pif[1]*Pfi[1] - Pif[2]*Pfi[2]) +1j*(Pif[1]*Pfi[2] + Pif[2]*Pfi[1]))
	g0n1 = coeff2*(Pif[3]*Pfi[1] -1j*(Pif[3]*Pfi[2]))
	g00 = Pif[3]*Pfi[3]
	g01 = -1*coeff2*(Pif[3]*Pfi[1] +1j*(Pif[3]*Pfi[2]))
	g1n1 = -1*coeff1*((Pif[1]*Pfi[1] - Pif[2]*Pfi[2]) -1j*(Pif[1]*Pfi[2] + Pif[2]*Pfi[1]))
	g10 = -1*coeff2*(Pif[1]*Pfi[3] -1j*(Pif[2]*Pfi[3]))
	g11 = coeff1*((Pif[1]*Pfi[1] + Pif[2]*Pfi[2]) +1j*(Pif[1]*Pfi[2] - Pif[2]*Pfi[1]))
	return np.array([[gn1n1,gn10,gn11],[g0n1,g00,g01],[g1n1,g10,g11]])

def ComboBoostMat(LI,LF):
	tmp = np.matmul(np.matmul(LF,gmn),LI)
	coeff1 = 1/2
	coeff2 = 1/math.sqrt(2)
	gn1n1 = coeff1*((tmp[1][1] + tmp[2][2]) -1j*(tmp[1][2] - tmp[2][1]))
	gn10 = coeff2*(tmp[1][3] +1j*(tmp[2][3]))
	gn11 = -1*coeff1*((tmp[1][1] - tmp[2][2]) +1j*(tmp[1][2] + tmp[2][1]))
	g0n1 = coeff2*(tmp[3][1] -1j*(tmp[3][2]))
	g00 = tmp[3][3]
	g01 = -1*coeff2*(tmp[3][1] +1j*(tmp[3][2]))
	g1n1 = -1*coeff1*((tmp[1][1] - tmp[2][2]) -1j*(tmp[1][2] + tmp[2][1]))
	g10 = -1*coeff2*(tmp[1][3] -1j*(tmp[2][3]))
	g11 = coeff1*((tmp[1][1] + tmp[2][2]) +1j*(tmp[1][2] - tmp[2][1]))
	return (1/4)*np.array([[gn1n1,gn10,gn11],[g0n1,g00,g01],[g1n1,g10,g11]])

def G_PwaveScalar(Pi,Pf,m1,m2,epsilon):
	si = dot(Pi, Pi)
	sf = dot(Pf, Pf)
	q = np.subtract(Pf, Pi)
	Q2 = -1*dot(q, q)
	qf = qStar(sf,m1,m2)
	qi = qStar(si,m1,m2)

	Lambda_I = BoostMatrixP(Pi)
	Lambda_F = BoostMatrixP(Pf)
	Pif = np.matmul(Lambda_F,Pi)
	Pfi = np.matmul(Lambda_I,Pf)
	MomMat = MomentumMat(Pif,Pfi)
	cbm = ComboBoostMat(Lambda_I,Lambda_F)
	
	int1 = I23(si,Q2,sf,m1,m2,epsilon)
	int2 = I24(si,Q2,sf,m1,m2,epsilon)

	return (3/(qi*qf))*(MomMat*int1 + cbm*int2)

#endregion