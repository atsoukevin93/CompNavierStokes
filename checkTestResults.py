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


def l2_error(U, dx, ax):
    return np.sqrt(np.sum(dx*U*U, axis=ax))


def l1_error(U, dx, ax):
    return np.sum(np.abs(dx*U), axis=ax)


def linf_error(U, ax):
    return np.max(np.abs(U), axis=ax)

nx = 250

# ------------------------- Code for Saving figures and generating gifs ----------------------------------------
dirpath_hkl = "figures/staggered_"+str(nx)+"_visco/"
filename_hkl = "staggered_"+str(nx)+"_visco"

# files = os.listdir(dirpath)
# files.sort()

numb_to_test = 1

data_hkl = np.load(dirpath_hkl+filename_hkl)

Rhos_hkl = data_hkl['Rho']
Us_hkl = data_hkl['U']

# --------------------- Time and Space steps ---------------------------------

dts = data_hkl['dt']
dxs = data_hkl['dx']
t = data_hkl['t']
Nt = t.shape[0]
Nvol = Rhos_hkl.shape[1]
dx = float(dxs[0])
dt = float(dxs[0])

# -------------------- data_constant_time_step for the rusanov method ----------------------------
dirpath_rusanov = "figures/rusanov_splitting_"+str(nx)+"_visco/"
filename_rusanov = "rusanov_splitting"+str(nx)+"_visco"
# filename_rusanov = "rusanov_splitting_test_new"

data_rusanov = np.load(dirpath_rusanov+filename_rusanov)

Rhos_rusanov = data_rusanov['Rho']
Us_rusanov = data_rusanov['U']

# Données du maillage
# Centres des mailles
mesh = PeriodicGrid1D(dx, Nvol)
x, = mesh.cellCenters
x_faces = mesh.faceCenters

# les inconnus
U_hkl = FaceVariable(name='$u$', mesh=mesh, value=0.)
# Rho0 = CellVariable(name='$\\rho$', mesh=mesh, value=0., hasOld=True)
Rho_hkl = CellVariable(name='$\\rho_{SSG}$', mesh=mesh, value=0., hasOld=True)
U_fig_hkl = CellVariable(name='$U_{SSG}$', mesh=mesh, value=0., hasOld=True)

Rho_rusanov = CellVariable(name='$\\rho_{rus}$', mesh=mesh, value=0., hasOld=True)
U_rusanov = CellVariable(name='$U_{rus}$', mesh=mesh, value=0., hasOld=True)


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
plt.rcParams['font.weight'] = 'bold'
plt.rcParams.update({'font.size': 15})

sp1, axes = plt.subplots(1,2)

Rho_fig = Matplotlib1DViewer(vars=(Rho_hkl, Rho_rusanov), axes=axes[0], interpolation='spline16', datamax=1.5, figaspect='auto', linewidth=4, legend= None)

u_fig = Matplotlib1DViewer(vars=(U_fig_hkl, U_rusanov), axes=axes[1], interpolation='spline16', datamax=0.15, figaspect='auto', linewidth=4, legend= None)

viewers = MultiViewer(viewers=(Rho_fig, u_fig))

# test_dir_path = "figures/hkl_vs_rusanov_tests/hkl_vs_rusanov_test_new/"
test_dir_path = "figures/staggered_vs_rusanov_visco/"

if not os.path.exists(test_dir_path):
    os.makedirs(test_dir_path)


for i in range(Nt):
    Rho_hkl.setValue(Rhos_hkl[i])
    U_hkl.setValue(Us_hkl[i])
    U_fig_hkl.setValue((U_hkl + shiftg(U_hkl))[0:Nvol] / 2.)

    Rho_rusanov.setValue(Rhos_rusanov[i])
    # U_rusanov.setValue((Us_rusanov[i] + shiftg(Us_rusanov[i]))[0:Nvol] / 2.)
    U_rusanov.setValue(Us_rusanov[i])

    viewers.plot()
    plt.title("SSG vs RUS tps={0}".format(str(round(t[i], 2))))

    if i % 2 == 0:
        plt.savefig(test_dir_path+'ssg_vs_rus_test_tps_{0}.png'.format(i))
    # plt.close()


# ------------------------------------- Error between Hkl and Rusanov Method -------------------------------------
dirpath = "data_new_adaptive_time_step/"
Num_method = ['rusanov_splitting', 'Staggered']
files = os.listdir(dirpath)
Nfiles = len(files)
file_suffix = '_NS_test_'
hkl_files = []
rusanov_files = []

L = 1

def get_nsteps(file_tab):
    N = len(file_tab)
    file_tab = np.array(file_tab)
    temp = np.empty([], dtype=np.int)
    temp = np.delete(temp, 0)

    for i in range(N):
        num = int(file_tab[i].split('_')[-1])
        temp = np.append(temp, num)
    temp.sort()
    return temp


def sort_files(file_tab):
    N = len(file_tab)
    file_tab = np.array(file_tab)
    temp = np.empty([], dtype=np.int)
    temp = np.delete(temp, 0)

    for i in range(N):
        num = int(file_tab[i].split('_')[-1])
        temp = np.append(temp, num)
    temp.sort()

    temp1 = np.empty([], dtype=np.str)
    temp1 = np.delete(temp1, 0)
    for i in range(N):
        for j in range(N):
            if int(file_tab[j].split('_')[-1]) == temp[i]:
                temp1 = np.append(temp1, file_tab[j])

    return temp1


for i in range(Nfiles):
    if files[i].startswith('rusanov'):
        rusanov_files.append(files[i])
    if files[i].startswith('Staggered'):
        hkl_files.append(files[i])

rusanov_files = sort_files(rusanov_files)
hkl_files = sort_files(hkl_files)
Ntest = len(rusanov_files)

rho_L2_norms = np.array([])
u_L2_norms = np.array([])
# limit = 50000
for i in range(Ntest):

    data_rusanov = np.load(dirpath + Num_method[0]+file_suffix + rusanov_files[i].split('_')[-1])
    dts = data_rusanov['dt']
    dxs = data_rusanov['dx']
    t = data_rusanov['t']
    Nt = t.shape[0]
    dx = float(dxs[0])
    Rhos_rusanov = data_rusanov['Rho']
    Nvol = Rhos_rusanov.shape[1]
    Us_rusanov = data_rusanov['U']
    # Us_rusanov.shape


    data_hkl = np.load(dirpath + Num_method[1]+file_suffix + hkl_files[i].split('_')[-1])
    Rhos_hkl = data_hkl['Rho']
    Us_hkl = data_hkl['U']
    Us_hkl_new = np.zeros((Us_hkl.shape[0], Us_hkl.shape[1]-1))

    # Us_hkl_new_reduced = np.zeros((Us_rusanov.shape[0], Us_rusanov.shape[1]))
    # Rhos_hkl_reduced = np.zeros((Rhos_rusanov.shape[0], Rhos_rusanov.shape[1]))

    for i in range(Nt):
        Us_hkl_new[i] = (Us_hkl[i] + shiftg(Us_hkl[i]))[0:Nvol]/2.
        # if i % 2 == 0:
        #     Us_hkl_new_reduced[i] = Us_hkl_new[i]
        #     Rhos_hkl_reduced[i] = Rhos_hkl[i]

    rho_L2_norms = np.append(rho_L2_norms, l2_error(l2_error(Rhos_hkl - Rhos_rusanov, dx, 1), dts, 0))
    u_L2_norms = np.append(u_L2_norms, l2_error(l2_error(Us_hkl_new-Us_rusanov, dx, 1), dts, 0))

hs = L/get_nsteps(rusanov_files)
hs.sort()

plt.rcParams['font.weight'] = 'bold'
plt.rcParams.update({'font.size': 18})

plt.figure(1, figsize=[10, 10])
plt.title('$L^2_t L^2_x$ error SSG vs RUS')
plt.xlabel('$\Delta_x$')
# plt.ylabel('$\Vert \\rho_{hkl} - \\rho_{rus} \Vert_{L^2_t L^2_x}$')
# plt.plot(hs, rho_L2_norms[np.arange(Ntest - 1, -1, -1)], linewidth=3, label='$\\rho$')
plt.plot(hs, rho_L2_norms[np.arange(Ntest - 1, -1, -1)], '*', linewidth=2, color='blue', label='$\Vert \\rho_{hkl} - \\rho_{rus} \Vert_{L^2_t L^2_x}$', )
# plt.legend()
# plt.savefig('rusanov_vs_staggered.png')

plt.title('$L^2_t L^2_x$ error SSG vs RUS')
plt.xlabel('$\Delta_x$')
# plt.ylabel('$\Vert u_{hkl} - u_{rus} \Vert_{L^2_t L^2_x}$')
# plt.plot(hs, u_L2_norms[np.arange(Ntest - 1, -1, -1)], linewidth=3, label='$u$')
plt.plot(hs, u_L2_norms[np.arange(Ntest - 1, -1, -1)], 'o', linewidth=2, color='red', label='$\Vert u_{hkl} - u_{rus} \Vert_{L^2_t L^2_x}$')
plt.legend()
plt.savefig('rusanov_vs_staggered2.png')


mesh = PeriodicGrid1D(dx, Nvol)
x, = mesh.cellCenters
x_faces = mesh.faceCenters

plt.title('$\\rho$ Hkl vs $\\rho$ rusanov at t=0.1')
plt.plot(x, Rhos_hkl[Nt -1], linewidth=3, label='$\\rho_{hkl}$')
plt.plot(x, Rhos_rusanov[Nt-1], linewidth=3, label='$\\rho_{rus}$')
plt.legend()


# les inconnus
# U_hkl = FaceVariable(name='$u$', mesh=mesh, value=0.)
U_hkl = CellVariable(name='$u$', mesh=mesh, value=0.)
# Rho0 = CellVariable(name='$\\rho$', mesh=mesh, value=0., hasOld=True)
Rho_hkl = CellVariable(name='$\\rho_{hkl}$', mesh=mesh, value=0., hasOld=True)
U_fig_hkl = CellVariable(name='$U_{hkl}$', mesh=mesh, value=0., hasOld=True)

Rho_rusanov = CellVariable(name='$\\rho_{rus}$', mesh=mesh, value=0., hasOld=True)
U_rusanov = CellVariable(name='$U_{rus}$', mesh=mesh, value=0., hasOld=True)

sp1, axes = plt.subplots(1, 2)

Rho_fig = Matplotlib1DViewer(vars=(Rho_hkl, Rho_rusanov), axes=axes[0], interpolation='spline16', datamax=1.5, figaspect='auto')

u_fig = Matplotlib1DViewer(vars=(U_fig_hkl, U_rusanov), axes=axes[1], interpolation='spline16', figaspect='auto')

viewers = MultiViewer(viewers=(Rho_fig, u_fig))

for i in range(Nt):
    Rho_hkl.setValue(Rhos_hkl[i])
    U_hkl.setValue(Us_hkl_new[i])
    U_fig_hkl.setValue((U_hkl + shiftg(U_hkl))[0:Nvol] / 2.)
    # U_fig_hkl.setValue(Us_hkl_new_reduced[i])

    Rho_rusanov.setValue(Rhos_rusanov[i])
    U_rusanov.setValue(Us_rusanov[i])

    viewers.plot()
    plt.title("HKL vs Rusanov. tps={0}".format(str(round(t[i], 2))))

    # if i % 50 == 0:
    #     plt.savefig(test_dir_path+'hkl_vs_rusanov_test_new_tps{0}.png'.format(i))
    # plt.close()
plt.plot(x , Us_rusanov[Nt-1], label="rusanov");
plt.plot(x , Us_hkl_new[Nt-1], label="hkl")
plt.legend()

# --------------------------------------L2 error---------------------------------------
# error_l2 = l2_error(Rhos_hkl-Rhos_rusanov[1:Nt+1], dx, 1)
# error_l1 = l1_error(Rhos_hkl-Rhos_rusanov[1:Nt+1], dx, 1)
# error_linf = linf_error(Rhos_hkl-Rhos_rusanov[1:Nt+1], 1)

#
# plt.plot(t, error_l1)
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

# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams.update({'font.size': 12})
# plt.title("initial data $\\rho_0 (x)$"); plt.xlabel("$\\rho_0(x)$"); plt.ylabel("$x$"); plt.plot(x , U1.value, linewidth=3);
