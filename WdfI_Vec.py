#! /usr/bin/env python3

# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
import functions_2 as func
import sys
import numpy as np
import pickle
# Define variables and set defaults
#region


m1 = 1
m1Name = 'defalut'
m2 = 1
m2Name = 'defalut'
g = 3
B = 0.1
mr = 2.2
Q2 = 1
xi = 0.5
eiFactor = 1.05
Ffactor = 0
sheet = 1

start = 0.9
end = 2.5
imagPart = 0
efSize = 50

showPlots = True
scaleSet = False
eieqef = False

verbose = False

save = False
saveName = 'a'

M1Data = {'full':np.array([]), 'real':np.array([]), 'imag':np.array([]), 'abs':np.array([])}
AData = {'full':np.array([]), 'real':np.array([]), 'imag':np.array([]), 'abs':np.array([])}
fData = np.array([])
GData = {'full':np.array([]), 'real':np.array([]), 'imag':np.array([]), 'abs':np.array([])}
M2Data = {'full':np.array([]), 'real':np.array([]), 'imag':np.array([]), 'abs':np.array([])}
WdfData = {'full':np.array([]), 'real':np.array([]), 'imag':np.array([]), 'abs':np.array([])}
WdfData_test = {'full':np.array([]), 'real':np.array([]), 'imag':np.array([]), 'abs':np.array([])}


sfRange = []
mrhoData = []
# rhoWrhoData = []
# FfGData = []

#endregion

# Parse command line input
#region

def setMass1(arg):
	global m1, m1Name
	m1 = masses[arg]
	m1Name = arg

def setMass2(arg):
	global m2, m2Name
	m2 = masses[arg]
	m2Name = arg

def setResonance(arg):
	global mr
	mr = masses[arg]

def setFfactor(arg):
	global Ffactor
	Ffactor = float(arg)

def setEi(arg):
	global eiFactor, eieqef
	if arg == 'sf':
		eieqef = True
		print("Setting si=sf")
	else:
		eiFactor = float(arg)

def setg(arg):
	global g
	g = float(arg)

def setB(arg):
	global B
	B = float(arg)

def setScale(arg):
	global scale, scaleSet
	scaleSet = True
	scale = masses[arg]

def setNumPoints(arg):
	global efSize
	efSize = int(arg)

def setStart(arg):
	global start
	start = float(arg)

def setEnd(arg):
	global end
	end = float(arg)

def setShowPlots(arg):
	global showPlots
	if arg == 'False':
		showPlots = False

def setImag(arg):
	global imagPart
	imagPart = float(arg)

def savePlot(arg):
	global saveName, save
	saveName = arg
	save = True

def setSheet(arg):
	global sheet
	tmp = int(arg)
	if tmp > 0 and tmp < 3:
		sheet = tmp

masses = {'pion':0.140, 'rho':0.777, 'proton':0.940, 'delta':1.23, 'test':2.2, 'unit':1.0}
options = {'-m1':setMass1, '-m2':setMass2, '-mr':setResonance, '-ei':setEi, '-g':setg, '-B':setB, 
			'-sc':setScale, '-N':setNumPoints, '-start':setStart, '-end':setEnd, '-A':setFfactor,
			'-show':setShowPlots, '-v':'verbose', '-imag':setImag, '-save':savePlot, '-sheet':setSheet}

for i,arg in enumerate(sys.argv):
	out = options.get(arg,'invalid')
	if out == 'verbose':
		verbose = True
	elif out != 'invalid':
		out(sys.argv[i+1])

if m1 != m2:
	xi = 1

eth = m1+m2
sth = pow(eth,2)
ei = eiFactor*eth
Pi = np.array([ei,0,0,0])

# print("scaleSet: " + str(scaleSet))

if not scaleSet:
	scale = pow(eth,2)
	# print("scale: " + str(scale))

if imagPart != 0:
	start = complex(start,imagPart)
	end = complex(end,imagPart)

de = (end-start)/efSize
efRange = np.linspace(start,end,efSize)
# efRange = [2.1444444444444]
# A=[0.085]

if verbose:
	print("*************** Run Parameters ***************")
	print("m1: " + str(m1))
	print("m2: " + str(m2))
	print("mr: " + str(mr))
	if eieqef:
		print("ei: " + 'ef')
	else:
		print("ei: " + str(eiFactor))
	print("g: " + str(g))
	print("A scale: " + str(B))
	print("scale: " + str(scale))
	print("N: " + str(efSize))
	print("ef start: " + str(start))
	print("ef end: " + str(end))
	print("sheet: " + str(sheet))
	print("")
	print("***********************************************")
	print("")

#endregion

# File opening
#region

# fileNameMrho = "Mrho_" + m1Name + "_" + m2Name + "_" + str(eiFactor) + "_" + str(g) + "_" + str(start) + "_" + str(end) + "_" + str(efSize) + ".txt"
fileNameG = "G_" + m1Name + "_" + m2Name + "_" + str(eiFactor) + "_" + str(Q2) + "_" + str(start) + "_" + str(end) + "_" + str(efSize) + "_" + str(sheet) + ".dat"
fileNameA = "A_" + m1Name + "_" + m2Name + "_" + str(eiFactor) + "_" + str(Q2) + "_" + str(start) + "_" + str(end) + "_" + str(efSize) + "_" + str(B) + ".dat"
fileNameWdf = "Wdf_" + m1Name + "_" + m2Name + "_" + str(eiFactor) + "_" + str(Ffactor) + "_" + str(g) + "_" + str(Q2) + "_" + str(start) + "_" + str(end) + "_" + str(efSize) + "_" + str(sheet) + ".txt"
# fileNameWCheck = "WCheck_" + m1Name + "_" + m2Name + "_" + str(eiFactor) + "_" + str(Ffactor) + "_" + str(g) + "_" + str(Q2) + "_" + str(start) + "_" + str(end) + "_" + str(efSize) + ".txt"

fileGOpen = False
fileAOpen = False
fileMrhoOpen = False

try:
	with open("../dataVec/"+fileNameG,"rb") as fg:
		GData = pickle.load(fg)
		fileGOpen = True
		print("Using G data from ../data/" + fileNameG)
except FileNotFoundError:
	print(fileNameG + " not found!")
	print("A new one will be created. This may increase run time.")

try:
	with open("../dataVec/"+fileNameA,"rb") as fa:
		AData = pickle.load(fa)
		fileAOpen = True
		print("Using A data from ../data/" + fileNameA)
except FileNotFoundError:
	print(fileNameA + " not found!")
	print("A new one will be created. This may increase run time.")

# try:
# 	fmp = open("./vecdata/"+fileNameMrho,"r")
# 	fileMrhoOpen = True
# 	fmp.close()
# except FileNotFoundError:
# 	print(fileNameMrho + " not found!")
# 	print("A new one will be created. This may increase run time.")

#endregion

# Calculate data
#region

# efTest = [1.5,2.5]
# sfimag_test = 0.4794

mg1 = func.M(pow(1.1,2),m1,m2,xi,mr,g)
mg2 = func.M(pow(1.2,2),m1,m2,xi,mr,g)

for i,ef in enumerate(efRange):

	Pf = np.array([ef+1j*imagPart,0,0,0])
	if eieqef:
		Pi = Pf

	si = func.dot(Pi,Pi)
	sf = func.dot(Pf,Pf)
	q = np.subtract(Pf,Pi)
	Q2 = -1*func.dot(q,q)
	sfRange.append(sf)

	if sheet == 1:
		M1 = func.M(si,m1,m2,xi,mr,g)
	elif sheet == 2:
		M1 = func.MII(si,m1,m2,xi,mr,g)
	M1Data['full'] = np.append(M1Data['full'], M1)
	M1Data['real'] = np.append(M1Data['real'], M1.real)
	M1Data['imag'] = np.append(M1Data['imag'], M1.imag)
	M1Data['abs'] = np.append(M1Data['abs'], abs(M1))

	if not fileAOpen:
		# A1 = func.Avector(B,Pi,Pf,m1,m2,mr,g,xi,de,0)
		# atest= abs(A1)
		A = func.Avector_2(B,Pi,Pf,m1,mr,g)
		AData['full'] = np.append(AData['full'], A[0])
		AData['real'] = np.append(AData['real'], A[0].real)
		AData['imag'] = np.append(AData['imag'], A[0].imag)
		AData['abs'] = np.append(AData['abs'], abs(A[0]))
	else:
		A = AData['full'][i]

	f = func.f(mr,Q2)
	fData = np.append(fData, f)

	if not fileGOpen:
		if sheet == 1:
			G = func.Gvector(Pi,Pf,m1,m2,0)
		elif sheet == 2:
			G = func.Gvector_II(Pi,Pf,m1,m2,0)
	
		GData['full'] = np.append(GData['full'], G[0])
		GData['real'] = np.append(GData['real'], G[0].real)
		GData['imag'] = np.append(GData['imag'], G[0].imag)
		GData['abs'] = np.append(GData['abs'], abs(G[0]))
	else:
		G = GData['full'][i]
	

	if sheet == 1:
		M2 = func.M(sf,m1,m2,xi,mr,g)
	elif sheet == 2:
		M2 = func.MII(sf,m1,m2,xi,mr,g)
	M2Data['full'] = np.append(M2Data['full'], M2)
	M2Data['real'] = np.append(M2Data['real'], M2.real)
	M2Data['imag'] = np.append(M2Data['imag'], M2.imag)
	M2Data['abs'] = np.append(M2Data['abs'], abs(M2))

	# wtmp = M1*(A+f*G)*M2
	# WdfData_test['full'] = np.append(WdfData_test['full'], wtmp[0])
	# WdfData_test['real'] = np.append(WdfData_test['real'], wtmp[0].real)
	# WdfData_test['imag'] = np.append(WdfData_test['imag'], wtmp[0].imag)
	# WdfData_test['abs'] = np.append(WdfData_test['abs'], abs(wtmp[0]))


	rhof = func.rho(sf,m1,m2,xi)
	mrhoData.append(abs(rhof*M2))


# print("A data")
# print(AData['full'][0])
# print(AData['full'])
# print()
# print("G data")
# print(GData['full'][0])
# print(GData['full'])
# print()
# print("f data")
# print(fData)

ff2 = AData['full']+fData*GData['full']
Wdf = M1Data['full']*ff2*M2Data['full']
WdfData['full'] = np.append(WdfData['full'], Wdf)
WdfData['real'] = np.append(WdfData['real'], Wdf.real)
WdfData['imag'] = np.append(WdfData['imag'], Wdf.imag)
WdfData['abs'] = np.append(WdfData['abs'], abs(Wdf))

# rhoWrhoData.append(abs(rhof*Wdf*rhof))
# FfGData.append(abs(ff2))


print("scale: " + str(scale))
#endregion

# File writing
#region

if not fileGOpen:
	with open("../dataVec/"+fileNameG,"w+b") as fg:
		pickle.dump(GData,fg)

if not fileAOpen:
	with open("../dataVec/"+fileNameA,"w+b") as fa:
		pickle.dump(AData,fa)

with open("../dataVec/"+fileNameWdf,"w+b") as fwdf:
	pickle.dump(Wdf,fwdf)

#endregion

# Plotting
#region

mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Futura'
mpl.rcParams['mathtext.it'] = 'Futura:italic'
mpl.rcParams['mathtext.bf'] = 'Futura:bold'


fig = plt.figure(figsize=(16, 10))

plot0 = fig.add_subplot(2,1,1)
# plot0.plot(efRange, WdfData['real'], linewidth=5)
plot0.plot(efRange, GData['real'], linewidth=5)
# plot0.set_ylabel(r'$Re\left(\mathcal{W}_{df}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
plot0.set_ylabel(r'$Re\left(\mathcal{G}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
plt.axvline(eth/scale,color='black')
# plt.ylim(-5000,5000)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
plt.text(0.9,0.9,r'$Im\left(E_{f}\right)=$'+str(imagPart),transform=plot0.transAxes,size=15)
if eieqef and sheet == 1:
	plt.title(r'$E_{i} = E_{f}$, Sheet I',size=20)
elif not eieqef and sheet == 1:
	plt.title(r'$E_{i}=$'+str(ei)+', Sheet I',size=20)
elif eieqef and sheet == 2:
	plt.title(r'$E_{i} = E_{f}$, Sheet II',size=20)
elif not eieqef and sheet == 2:
	plt.title(r'$E_{i}=$'+str(ei)+', Sheet II',size=20)

plot1 = fig.add_subplot(2,1,2)
# plot1.plot(efRange, WdfData['imag'], linewidth=5)
plot1.plot(efRange, GData['imag'], linewidth=5)
plot1.set_xlabel(r'$E_{f}$',fontname="Futura",size=15)
# plot1.set_ylabel(r'$Im\left(\mathcal{W}_{df}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
plot1.set_ylabel(r'$Im\left(\mathcal{G}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
plt.axvline(eth/scale,color='black')
# plt.ylim(-5000,5000)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)

if save:
	plt.savefig(saveName, dpi=200)

if showPlots:
	plt.show()

#endregion

























