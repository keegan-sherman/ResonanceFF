#! /usr/bin/env python

import tri_lib
import numpy as np
import matplotlib.pyplot as plt

m1 = 1.0
m2 = 1.0
Q2 = 0.5
eth = m1+m2
sth = pow(eth,2)
si = 1.005*sth
sfRange = np.linspace(3.6,5.0,100)

realG_1 = []
imagG_1 = []
# realG_2 = []
# imagG_2 = []

G = tri_lib.G0(si,Q2,4.5,m1,m2,0)
print(f"real: {G[0][0]}")
print("real Info:")
print(G[0][2])
print()

realTest = sum(G[0][2]['rlist'])

print(realTest)

# print(f"imag: {G[1][0]}")
# print("imag Info:")
# print(G[1][2])


# for sf in sfRange:
#     G = tri_lib.G0(si,Q2,sf,m1,m2,0)
#     realG_1.append(G[0])
#     imagG_1.append(G[1])
#     # realG_2.append(G[2].real)
#     # imagG_2.append(G[2].imag)


# fig1 = plt.figure(figsize=(15,9))

# plotReal = fig1.add_subplot(2,1,1)
# plotReal.plot(sfRange,realG_1)
# # plotReal.plot(sfRange,realG_2)
# plotImag = fig1.add_subplot(2,1,2)
# plotImag.plot(sfRange,imagG_1)
# # plotImag.plot(sfRange,imagG_2)
# plt.show()
