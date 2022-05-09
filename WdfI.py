#! /usr/bin/env python3

# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib as mpl
import functions_2 as func
import numpy as np
import math
import sys
import argparse as ap
import pickle

# Parse command line input and set defaults
#region

masses = {'pion':0.140, 'rho':0.777, 'proton':0.940, 'delta':1.23, 'test':2.2, 'unit':1.0}
# options = {'-m1':setMass1, '-m2':setMass2, '-mr':setResonance, '-si':setSi, '-g':setg, '-Q2':setQ2, 
			# '-sc':setScale, '-N':setNumPoints, '-start':setStart, '-end':setEnd, '-A':setFfactor,
			# '-show':setShowPlots, '-v':'verbose', '-imag':setImag, '-save':savePlot, '-sheet':setSheet}

parser = ap.ArgumentParser()
parser.add_argument('-m1',default='unit',type=str)
parser.add_argument('-m2',default='unit',type=str)
parser.add_argument('-mr',default='test',type=str)
parser.add_argument('-ei',default='1.05',type=str)
parser.add_argument('-g',default=3.0,type=float)
parser.add_argument('-Q2',default=1.0,type=float)
parser.add_argument('-sc','--scale',default=False,type=bool)
parser.add_argument('-N',default=50,type=int)
parser.add_argument('-start',default=0.9,type=float)
parser.add_argument('-end',default=2.5,type=float)
parser.add_argument('-A',default=0.1,type=float)
parser.add_argument('-show',default=True,type=bool)
parser.add_argument('-v','--verbose',action='store_true')
parser.add_argument('-imag',default=0,type=float)
parser.add_argument('-save',default='notSaved',type=str)
parser.add_argument('-sheet',default=1,type=int)

args = vars(parser.parse_args())
print(args)

m1Name = args['m1']
m1 = masses[m1Name]
m2Name = args['m2']
m2 = masses[m2Name]
g = args['g']
mr = masses[args['mr']]
Q2 = args['Q2']
Ffactor = args['A']
sheet = args['sheet']
start = args['start']
end = args['end']
imagPart = args['imag']
efSize = args['N']
showPlots = args['show']
verbose = args['verbose']

if args['ei'] == 'sf':
	sieqsf = True
	print("Setting si=sf")
else:
	sieqsf = False
	eiFactor = float(args['ei'])

eth = m1+m2
ei = eiFactor*eth
si = pow(ei,2)

if args['save'] != 'notSaved':
	save = True
	saveName = args['save']
else:
	save = False

if m1 == m2:
	xi = 0.5
else:
	xi = 1

if args['scale']:
	scale = eth
else:
	scale = 1

if imagPart != 0:
	start = complex(start,imagPart)
	end = complex(end,imagPart)

efRange = np.linspace(start,end,efSize)

if args['verbose']:
	print("*************** Run Parameters ***************")
	print("m1: " + str(m1))
	print("m2: " + str(m2))
	print("mr: " + str(mr))
	if sieqsf:
		print("ei: " + 'sf')
	else:
		print("ei: " + str(eiFactor))
	print("g: " + str(g))
	print("Q2: " + str(Q2))
	print("A scale: " + str(Ffactor))
	print("scale: " + str(scale))
	print("N: " + str(efSize))
	print("sf start: " + str(start))
	print("sf end: " + str(end))
	print("sheet: " + str(sheet))
	print("")
	print("***********************************************")
	print("")

M1Data = {'full':np.array([]), 'real':np.array([]), 'imag':np.array([]), 'abs':np.array([])}
AData = {'full':np.array([]), 'real':np.array([]), 'imag':np.array([]), 'abs':np.array([])}
fData = np.array([])
GData = {'full':np.array([]), 'real':np.array([]), 'imag':np.array([]), 'abs':np.array([])}
M2Data = {'full':np.array([]), 'real':np.array([]), 'imag':np.array([]), 'abs':np.array([])}
WdfData = {'full':np.array([]), 'real':np.array([]), 'imag':np.array([]), 'abs':np.array([])}

rhoWrhoData = []
FfGData = []
#endregion

# File opening
#region

fileNameMrho = "Mrho_" + m1Name + "_" + m2Name + "_" + str(eiFactor) + "_" + str(g) + "_" + str(start) + "_" + str(end) + "_" + str(efSize) + ".txt"
fileNameG = "G_" + m1Name + "_" + m2Name + "_" + str(eiFactor) + "_" + str(Q2) + "_" + str(start) + "_" + str(end) + "_" + str(efSize) + "_" + str(sheet) + ".txt"
fileNameWdf = "Wdf_" + m1Name + "_" + m2Name + "_" + str(eiFactor) + "_" + str(Ffactor) + "_" + str(g) + "_" + str(Q2) + "_" + str(start) + "_" + str(end) + "_" + str(efSize) + "_" + str(sheet) + ".txt"
fileNameWCheck = "WCheck_" + m1Name + "_" + m2Name + "_" + str(eiFactor) + "_" + str(Ffactor) + "_" + str(g) + "_" + str(Q2) + "_" + str(start) + "_" + str(end) + "_" + str(efSize) + ".txt"

fileGOpen = False
fileMrhoOpen = False

try:
	with open("./data/"+fileNameG,"rb") as fg:
		GData = pickle.load(fg)
		fileGOpen = True
		print("Using G data from ./data/" + fileNameG)
except FileNotFoundError:
	print(fileNameG + " not found!")
	print("A new one will be created. This may increase run time.")

# try:
# 	fg = open("../data/"+fileNameG,"r")
# 	fileGOpen = True
# except FileNotFoundError:
# 	print(fileNameG + " not found!")
# 	print("A new one will be created. This may increase run time.")

# if fileGOpen:
# 	lines = fg.readlines()
# 	realGData = [float(i) for i in lines[0].rstrip("\n").split()]
# 	imagGData = [float(i) for i in lines[1].rstrip("\n").split()]
# 	absGData = [abs(complex(i)) for i in lines[2].rstrip("\n").split()]

# try:
# 	fmp = open("../data/"+fileNameMrho,"r")
# 	fileMrhoOpen = True
# 	fmp.close()
# except FileNotFoundError:
# 	print(fileNameMrho + " not found!")
# 	print("A new one will be created. This may increase run time.")

#endregion

# Calculate data
#region

mg1 = func.M(pow(1.1,2),m1,m2,xi,mr,g)
mg2 = func.M(pow(1.2,2),m1,m2,xi,mr,g)

for i,ef in enumerate(efRange):
	sf = pow(ef,2)
	if sieqsf:
		si = sf

	if sheet == 1:
		M1 = func.M(si,m1,m2,xi,mr,g)
	elif sheet == 2:
		M1 = func.MII(si,m1,m2,xi,mr,g)
	M1Data['full'] = np.append(M1Data['full'],M1)
	M1Data['real'] = np.append(M1Data['real'],M1.real)
	M1Data['imag'] = np.append(M1Data['imag'],M1.imag)
	M1Data['abs'] = np.append(M1Data['abs'],abs(M1))

	A = func.F(mr,Q2,g,Ffactor)
	AData['full'] = np.append(AData['full'],A)
	AData['real'] = np.append(AData['real'],A.real)
	AData['imag'] = np.append(AData['imag'],A.imag)
	AData['abs'] = np.append(AData['abs'],abs(A))

	f = func.f(mr,Q2)
	fData = np.append(fData,f)

	if not fileGOpen:
		if sheet == 1:
			real, imag = func.G0(si,Q2,sf,m1,m2, 0)
		elif sheet == 2:
			real, imag = func.G0II(si,Q2,sf,m1,m2, 0)
		G = complex(real,imag)
		GData['full'] = np.append(GData['full'],G)
		GData['real'] = np.append(GData['real'],real)
		GData['imag'] = np.append(GData['imag'],imag)
		GData['abs'] = np.append(GData['abs'],abs(G))

	if sheet == 1:
		M2 = func.M(sf,m1,m2,xi,mr,g)
	elif sheet == 2:
		M2 = func.MII(sf,m1,m2,xi,mr,g)
	M2Data['full'] = np.append(M2Data['full'],M2)
	M2Data['real'] = np.append(M2Data['real'],M2.real)
	M2Data['imag'] = np.append(M2Data['imag'],M2.imag)
	M2Data['abs'] = np.append(M2Data['abs'],abs(M2))

	# rhof = func.rho(sf,m1,m2,xi)
	# np.append(mrhoData,abs(rhof*M2))

ff2 = AData['full']+fData*GData['full']
Wdf = M1Data['full']*ff2*M2Data['full']
WdfData['full'] = np.append(WdfData['full'],Wdf)
WdfData['real'] = np.append(WdfData['real'],Wdf.real)
WdfData['imag'] = np.append(WdfData['imag'],Wdf.imag)
WdfData['abs'] = np.append(WdfData['abs'],abs(Wdf))


if args['scale']:
	efRange = np.divide(efRange,scale)
	print("scale: " + str(scale))
#endregion

# File writing
#region

if not fileGOpen:
	with open("./data/"+fileNameG,"w+b") as fg:
		pickle.dump(GData,fg)

with open("./data/"+fileNameWdf,"w+b") as fwdf:
	pickle.dump(WdfData,fwdf)

# if not fileGOpen:
# 	fg = open("../data/"+fileNameG,"w+")

# 	for i in realGData:
# 		fg.write(str(i) + " ")
# 	fg.write("\n")

# 	for i in imagGData:
# 		fg.write(str(i) + " ")
# 	fg.write("\n")

# 	for i in absGData:
# 		fg.write(str(i) + " ")

# fg.close()

# fwdf = open("../data/"+fileNameWdf,"w+")

# for i in realWdfData:
# 	fwdf.write(str(i) + " ")
# fwdf.write("\n")

# for i in imagWdfData:
# 	fwdf.write(str(i) + " ")
# fwdf.write("\n")

# for i in absWdfData:
# 	fwdf.write(str(i) + " ")

# fwdf.close()

# if not fileMrhoOpen:
# 	fmp = open("../data/"+fileNameMrho,"w+")

# 	for i in mrhoData:
# 		fmp.write(str(i) + " ")
# 	fmp.close()

# fwc = open("../data/"+fileNameWCheck,"w+")

# for i in rhoWrhoData:
# 	fwc.write(str(i) + " ")
# fwc.write("\n")

# for i in FfGData:
# 	fwc.write(str(i) + " ")
# fwc.write("\n")

# fwc.close()

#endregion

# Plotting
#region

mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Futura'
mpl.rcParams['mathtext.it'] = 'Futura:italic'
mpl.rcParams['mathtext.bf'] = 'Futura:bold'


fig = plt.figure(figsize=(16, 10))

plot0 = fig.add_subplot(2,1,1)
plot0.plot(efRange, WdfData['real'], linewidth=5)
plot0.set_ylabel(r'$Re\left(\mathcal{W}_{df}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
# plot0.set_ylabel(r'$Re\left(\mathcal{M}\right)$',size=15)
# plot0.set_ylabel(r'$Re\left(\mathcal{G}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
plt.axvline(eth/scale,color='black')
# plt.ylim(-700,1500)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)
plt.text(0.9,0.9,r'$Im\left(s_{f}\right)=$'+str(imagPart),transform=plot0.transAxes,size=15)
if sieqsf and sheet == 1:
	plt.title(r'$s_{i} = s_{f}$, Sheet I',size=20)
elif not sieqsf and sheet == 1:
	plt.title(r'$s_{i}=$'+str(ei)+', Sheet I',size=20)
elif sieqsf and sheet == 2:
	plt.title(r'$s_{i} = s_{f}$, Sheet II',size=20)
elif not sieqsf and sheet == 2:
	plt.title(r'$s_{i}=$'+str(si)+', Sheet II',size=20)

plot1 = fig.add_subplot(2,1,2)
plot1.plot(efRange, WdfData['imag'], linewidth=5)
plot1.set_xlabel(r'$e_{f}$',fontname="Futura",size=15)
plot1.set_ylabel(r'$Im\left(\mathcal{W}_{df}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
# plot1.set_ylabel(r'$Im\left(\mathcal{M}\right)$',size=15)
# plot1.set_ylabel(r'$Im\left(\mathcal{G}\right)\left(\frac{1}{GeV}^{2}\right)$',size=15)
plt.axvline(eth/scale,color='black')
# plt.ylim(-750,1300)
plt.xticks(fontname="Futura",fontsize=15)
plt.yticks(fontname="Futura",fontsize=15)

if save:
	plt.savefig(saveName, dpi=200)

if showPlots:
	plt.show()

#endregion

























