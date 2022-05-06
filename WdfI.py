#! /usr/bin/env python3

# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
import functions_2 as func
import numpy as np
import math
import sys

# Define variables and set defaults
#region


m1 = 1
m1Name = 'defalut'
m2 = 1
m2Name = 'defalut'
g = 3
mr = 2.5
Q2 = 1
xi = 0.5
siFactor = 1.1
Ffactor = 0
sheet = 1

start = 0.9
end = 2.5
imagPart = 0
sfSize = 50

showPlots = True
scaleSet = False
sieqsf = False

verbose = False

save = False
saveName = 'a'

realM1Data = []
imagM1Data = []
absM1Data = []

realFData = []
imagFData = []
absFData = []

realfData = []
imagfData = []
absfData = []

realGData = []
imagGData = []
absGData = []

mrhoData = []

realM2Data = []
imagM2Data = []
absM2Data = []

realWdfData = []
imagWdfData = []
absWdfData = []

rhoWrhoData = []
FfGData = []

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

def setSi(arg):
	global siFactor, sieqsf
	if arg == 'sf':
		sieqsf = True
		print("Setting si=sf")
	else:
		siFactor = float(arg)

def setg(arg):
	global g
	g = float(arg)

def setQ2(arg):
	global Q2
	Q2 = float(arg)

def setScale(arg):
	global scale, scaleSet
	scaleSet = True
	scale = masses[arg]

def setNumPoints(arg):
	global sfSize
	sfSize = int(arg)

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
options = {'-m1':setMass1, '-m2':setMass2, '-mr':setResonance, '-si':setSi, '-g':setg, '-Q2':setQ2, 
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

sth = pow(m1+m2,2)
si = siFactor*sth

# print("scaleSet: " + str(scaleSet))

if not scaleSet:
	scale = sth
	# print("scale: " + str(scale))

if imagPart != 0:
	start = complex(start,imagPart)
	end = complex(end,imagPart)

sfRange = np.linspace(start,end,sfSize)

if verbose:
	print("*************** Run Parameters ***************")
	print("m1: " + str(m1))
	print("m2: " + str(m2))
	print("mr: " + str(mr))
	if sieqsf:
		print("si: " + 'sf')
	else:
		print("si: " + str(siFactor))
	print("g: " + str(g))
	print("Q2: " + str(Q2))
	print("A scale: " + str(Ffactor))
	print("scale: " + str(scale))
	print("N: " + str(sfSize))
	print("sf start: " + str(start))
	print("sf end: " + str(end))
	print("sheet: " + str(sheet))
	print("")
	print("***********************************************")
	print("")

#endregion

# File opening
#region

fileNameMrho = "Mrho_" + m1Name + "_" + m2Name + "_" + str(siFactor) + "_" + str(g) + "_" + str(start) + "_" + str(end) + "_" + str(sfSize) + ".txt"
fileNameG = "G_" + m1Name + "_" + m2Name + "_" + str(siFactor) + "_" + str(Q2) + "_" + str(start) + "_" + str(end) + "_" + str(sfSize) + "_" + str(sheet) + ".txt"
fileNameWdf = "Wdf_" + m1Name + "_" + m2Name + "_" + str(siFactor) + "_" + str(Ffactor) + "_" + str(g) + "_" + str(Q2) + "_" + str(start) + "_" + str(end) + "_" + str(sfSize) + "_" + str(sheet) + ".txt"
fileNameWCheck = "WCheck_" + m1Name + "_" + m2Name + "_" + str(siFactor) + "_" + str(Ffactor) + "_" + str(g) + "_" + str(Q2) + "_" + str(start) + "_" + str(end) + "_" + str(sfSize) + ".txt"

fileGOpen = False
fileMrhoOpen = False

try:
	fg = open("../data/"+fileNameG,"r")
	fileGOpen = True
except FileNotFoundError:
	print(fileNameG + " not found!")
	print("A new one will be created. This may increase run time.")

if fileGOpen:
	lines = fg.readlines()
	realGData = [float(i) for i in lines[0].rstrip("\n").split()]
	imagGData = [float(i) for i in lines[1].rstrip("\n").split()]
	absGData = [abs(complex(i)) for i in lines[2].rstrip("\n").split()]

try:
	fmp = open("../data/"+fileNameMrho,"r")
	fileMrhoOpen = True
	fmp.close()
except FileNotFoundError:
	print(fileNameMrho + " not found!")
	print("A new one will be created. This may increase run time.")

#endregion

# Calculate data
#region

mg1 = func.M(pow(1.1,2),m1,m2,xi,mr,g)
mg2 = func.M(pow(1.2,2),m1,m2,xi,mr,g)

for i,sf in enumerate(sfRange):
	if sieqsf:
		si = sf

	if sheet == 1:
		M1 = func.M(si,m1,m2,xi,mr,g)
	elif sheet == 2:
		M1 = func.MII(si,m1,m2,xi,mr,g)
	realM1Data.append(M1.real)
	imagM1Data.append(M1.imag)
	absM1Data.append(abs(M1))

	F = func.F(mr,Q2,g,Ffactor)
	realFData.append(F.real)
	imagFData.append(F.imag)
	absFData.append(abs(F))

	f = func.f(mr,Q2)
	realfData.append(f.real)
	imagfData.append(f.imag)
	absfData.append(abs(f))

	if not fileGOpen:
		if sheet == 1:
			real, imag = func.G0(si,Q2,sf,m1,m2, 0)
		elif sheet == 2:
			real, imag = func.G0II(si,Q2,sf,m1,m2, 0)
		G = complex(real,imag)
		# G = func.GI(sf,m1,m2)
		# real = G.real
		# imag = G.imag
		realGData.append(real)
		imagGData.append(imag)
		absGData.append(abs(G))
	else:
		G = complex(realGData[i],imagGData[i])

	if sheet == 1:
		M2 = func.M(sf,m1,m2,xi,mr,g)
	elif sheet == 2:
		M2 = func.MII(sf,m1,m2,xi,mr,g)
	# M2 = func.M(sf,m1,m2,xi,mr,g)
	realM2Data.append(M2.real)
	imagM2Data.append(M2.imag)
	absM2Data.append(abs(M2))

	rhof = func.rho(sf,m1,m2,xi)
	mrhoData.append(abs(rhof*M2))

	ff2 = F+f*G

	Wdf = M1*ff2*M2
	# print("sf: " + str(sf) + ", ff2: " + str(ff2) + ", Wdf: " + str(Wdf))
	realWdfData.append(Wdf.real)
	imagWdfData.append(Wdf.imag)
	absWdfData.append(abs(Wdf))

	rhoWrhoData.append(abs(rhof*Wdf*rhof))
	FfGData.append(abs(ff2))

efRange = np.lib.scimath.sqrt(sfRange)
sfRange = np.divide(sfRange,scale)
print("scale: " + str(scale))
#endregion

# File writing
#region

if not fileGOpen:
	fg = open("../data/"+fileNameG,"w+")

	for i in realGData:
		fg.write(str(i) + " ")
	fg.write("\n")

	for i in imagGData:
		fg.write(str(i) + " ")
	fg.write("\n")

	for i in absGData:
		fg.write(str(i) + " ")

fg.close()

fwdf = open("../data/"+fileNameWdf,"w+")

for i in realWdfData:
	fwdf.write(str(i) + " ")
fwdf.write("\n")

for i in imagWdfData:
	fwdf.write(str(i) + " ")
fwdf.write("\n")

for i in absWdfData:
	fwdf.write(str(i) + " ")

fwdf.close()

if not fileMrhoOpen:
	fmp = open("../data/"+fileNameMrho,"w+")

	for i in mrhoData:
		fmp.write(str(i) + " ")
	fmp.close()

fwc = open("../data/"+fileNameWCheck,"w+")

for i in rhoWrhoData:
	fwc.write(str(i) + " ")
fwc.write("\n")

for i in FfGData:
	fwc.write(str(i) + " ")
fwc.write("\n")

fwc.close()

#endregion

# Plotting
#region

mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Futura'
mpl.rcParams['mathtext.it'] = 'Futura:italic'
mpl.rcParams['mathtext.bf'] = 'Futura:bold'


fig = plt.figure(figsize=(16, 10))

plot0 = fig.add_subplot(2,1,1)
plot0.plot(sfRange, realWdfData, linewidth=5)
plot0.set_ylabel(r'$Re\left(\mathcal{W}_{df}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
# plot0.set_ylabel(r'$Re\left(\mathcal{M}\right)$',size=15)
# plot0.set_ylabel(r'$Re\left(\mathcal{G}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
plt.axvline(sth/scale,color='black')
# plt.ylim(-700,1500)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
plt.text(0.9,0.9,r'$Im\left(s_{f}\right)=$'+str(imagPart),transform=plot0.transAxes,size=15)
if sieqsf and sheet == 1:
	plt.title(r'$s_{i} = s_{f}$, Sheet I',size=20)
elif not sieqsf and sheet == 1:
	plt.title(r'$s_{i}=$'+str(si)+', Sheet I',size=20)
elif sieqsf and sheet == 2:
	plt.title(r'$s_{i} = s_{f}$, Sheet II',size=20)
elif not sieqsf and sheet == 2:
	plt.title(r'$s_{i}=$'+str(si)+', Sheet II',size=20)

plot1 = fig.add_subplot(2,1,2)
plot1.plot(sfRange, imagWdfData, linewidth=5)
plot1.set_xlabel(r'$s_{f}$',fontname="Futura",size=15)
plot1.set_ylabel(r'$Im\left(\mathcal{W}_{df}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
# plot1.set_ylabel(r'$Im\left(\mathcal{M}\right)$',size=15)
# plot1.set_ylabel(r'$Im\left(\mathcal{G}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
plt.axvline(sth/scale,color='black')
# plt.ylim(-750,1300)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)

if save:
	plt.savefig(saveName, dpi=200)

if showPlots:
	plt.show()

#endregion

























