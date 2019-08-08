#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fipy import *
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sp;
import scipy.sparse.linalg as splin;
from scipy.optimize import newton
from scipy.optimize import fsolve
from ToolBox import *
# import rusanov.Rusanov as Ru
from Diffusion1D import *
import time


# Paramètre du maillage 1D
L = 1.0
nx =500
dx = L / nx


# Construction du maillage
mesh = PeriodicGrid1D(dx, nx)


# Données du maillage
# Centres des mailles
x, = mesh.cellCenters


# Nombre de volumes, nombre de faces
Nvol = mesh.numberOfCells
nFaces = mesh.numberOfFaces


# Faces associées aux cellules (en périodique)
FacesCells = mesh.faceCellIDs


# Paramètre p(rho)=c*rho^{gamma}
c = 1.
gamma = 3.

# les inconnus
U = FaceVariable(name='$u$', mesh=mesh, value=0.)
Rho = CellVariable(name='$\\rho$', mesh=mesh, value=0., hasOld=True)

# Donnée initiale sur rho
Rho.setValue(1., where=x >= 0.7)
Rho.setValue(1., where=x < 0.5)
Rho.setValue((np.sqrt(2.))**(1./gamma), where=(x > 0.5) & (x < 0.7))


#
#rho=Rho.value
#U1=U.value

U_fig = CellVariable(name='$U$', mesh=mesh, value=0., hasOld=True)


# Rho1.setValue(np.exp(-(x-0.5)**2)+0.3)

def mu_rho(rho):
    N=len(rho)
    return rho


def Explicit_Staggered(rho, U, dt, dx):
    N = len(rho)
    F = (U + shiftg(U)) / 2.
    G= (U+shiftd(U))/ 2.
    #equation de consevation de la masse
    diag0m = 1 -(dt/dx) * (pplus(U[0: N-1]) - pminus(U[0: N-1]))
    diag0= -(dt/dx) * (pminus(U[1:N-1]))
    diag0p=dt/dx * (pplus(U[0: N-1]))
    A = sp.lil_matrix(sp.spdiags([diag0m, diag0, diag0p], [-1, 0, 1], N, N))
    A[0, N-1] = dt/dx * (pplus(U[0]))
    A[N-1, 0] = -dt/dx *(pminus(U[N]))
    Rho_new= A.dot(Rho.value)

    # eq de conservation du moment, on l'écrit sous la forme AU=B
    diag0 = (1/(2*dt) * (shiftd(Rho_new)+ Rho_new) + 1/(dx*dx)*(mu_rho(Rho_new)+shiftd(mu_rho(Rho_new))))
    diag0p = np.concatenate([[0], (-1/(dx*dx)*mu_rho(Rho_new))[0:N]])
    diag0m = np.concatenate([1/(dx*dx) * mu_rho(Rho_new)[0:N], [0]])
    M = sp.lil_matrix(sp.spdiags([diag0m, diag0, diag0p], [-1, 0, 1], N + 1, N + 1))
    M[0, N] = -mu_rho(Rho_new)[0] / (dx*dx)
    M[N, 0] = mu_rho(Rho_new)[N - 1] /(dx*dx)

    B = 1/dt * (np.concatenate(shiftd(rho) + rho, (shiftd(rho)+ rho)[0]) * U
      -1/dx * np.concatenate([barotrope_rhotoP(Rho_new, c, gamma) -shiftd(barotrope_rhotoP(Rho_new, c, gamma)),
                            [barotrope_rhotoP(Rho_new, c, gamma) -shiftd(barotrope_rhotoP(Rho_new, c, gamma))[0]]])
      -1/dx * np.concatenate([rho, [rho[0]]]) * (U * pplus(F) + shiftg(U) * pminus(F))
      +1/dx * np.concatenate([shiftd(rho), [shiftd(rho)[0]]]) * (shiftd(U) * pplus(G) + U* pminus(G)))
    U_new = splin.spsolve(M,B)
    return Rho_new, U_new

# Boucle en temps
dt1 = 1e-4
duration = 100
Nt = int(duration / dt1) + 1
dt = dt1
tps = 0.

sp1, axes = plt.subplots(1,2)

Rho_fig = Matplotlib1DViewer(vars=Rho, axes=axes[0], interpolation='spline16', datamax = 1.5, figaspect='auto')

u_fig = Matplotlib1DViewer(vars=U_fig, axes=axes[1], interpolation='spline16', figaspect='auto')

viewers = MultiViewer(viewers=(Rho_fig, u_fig))

while tps <= duration:
    Rho_new , U_new = Explicit_Staggered(Rho.value, U.value, dt, dx)
    Rho.setValue(Rho_new)
    U.setValue(U_new)
    U_fig.setValue((U + shiftg(U))[0:Nvol]/2)

    viewers.plot()
