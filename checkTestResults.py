#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fipy import *
import numpy as np
import time
import sys
import os
from matplotlib import pyplot as plt
from ToolBox import *
import matplotlib.animation as animation

plt.ion()

dirpath = "data/"
filename = "hkl_test"

files = os.listdir(dirpath)
files.sort()

numb_to_test = 1

data = np.load(dirpath+filename)

dts = data['dt']
dxs = data['dx']
Rhos = data['Rho']
Us = data['U']
t = data['t']
Nt = t.shape[0]
Nvol = Rhos.shape[1]
dx = float(dxs[0])
# dt = float(dxs[0])


# Données du maillage
# Centres des mailles
mesh = PeriodicGrid1D(dx, Nvol)
x, = mesh.cellCenters
x_faces = mesh.faceCenters

# les inconnus
U = FaceVariable(name='$u$', mesh=mesh, value=0.)
# Rho0 = CellVariable(name='$\\rho$', mesh=mesh, value=0., hasOld=True)
Rho1 = CellVariable(name='$\\rho$', mesh=mesh, value=0., hasOld=True)
U_fig = CellVariable(name='$U$', mesh=mesh, value=0., hasOld=True)


# sp, spa = plt.subplots(1, 2)
# axes = spa.reshape(-1)
#
# for i in range(Nt):
#
#     if i == 0:
#         line1, = axes[0].plot(x, Rhos[i])
#
#         axes[1].set_ylim(-(np.max(((Us[i] + shiftg(Us[i])) / 2.)[0:Nvol]) + 0.0006),
#                          np.max(((Us[i] + shiftg(Us[i])) / 2.)[0:Nvol]) + 0.0006)
#         line2, = axes[1].plot(x, ((Us[i] + shiftg(Us[i]))/2.)[0:Nvol])
#     else:
#         line1.set_ydata(Rhos[i])
#         line2.set_ydata(((Us[i] + shiftg(Us[i]))/2.)[0:Nvol])
#     plt.pause(2)  # pause avec duree en secondes
#     t = t + dts[i]

sp1, axes = plt.subplots(1,2)

Rho_fig = Matplotlib1DViewer(vars=Rho1, axes=axes[0], interpolation='spline16', datamax = 1.5, figaspect='auto')

u_fig = Matplotlib1DViewer(vars=U_fig, axes=axes[1], interpolation='spline16', figaspect='auto')

viewers = MultiViewer(viewers=(Rho_fig, u_fig))

if not os.path.exists("figures/"):
    os.makedirs("figures/")

for i in range(Nt):
    Rho1.setValue(Rhos[i])
    U.setValue(Us[i])
    U_fig.setValue((U + shiftg(U))[0:Nvol] / 2.)
    viewers.plot()
    plt.title("HKL Method tps={0}".format(str(round(t[i], 2))))
    if i % 100 == 0:
        plt.savefig('figures/hkl_test_tps{0}.png'.format(i))
    # plt.close()



# fig = plt.figure()  # import matplotlib.animation as animationinitialise la figure


# fig, spa = plt.subplots(1, 2)
# axes = spa.reshape(-1)
# line1, = axes[0].plot([], [])
# line2, = axes[1].plot([], [])
# # plt.xlim(xmin, xmax)
# # plt.ylim(-1, 1)
#
#
# # fonction à définir quand blit=True
# # crée l'arrière de l'animation qui sera présent sur chaque image
# def init():
#     line1.set_data([], [])
#     line2.set_data([], [])
#     return line1, line2
#
#
# def animate(i):
#     tps = t[i]
#     line1.set_data(x, Rhos[i])
#     line2.set_data(x,((Us[i] + shiftg(Us[i]))/2.)[0:Nvol])
#     return line1, line2
#
#
# ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, blit=True, interval=20, repeat=False)
#
