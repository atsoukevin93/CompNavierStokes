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
nx =100
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
gamma = 2.

# les inconnus
U = FaceVariable(name='$u$', mesh=mesh, value=0.)
Rho = CellVariable(name='$\\rho$', mesh=mesh, value=0., hasOld=True)

# Donnée initiale sur rho
Rho.setValue(1., where=x >= 0.7)
Rho.setValue(1., where=x < 0.5)
Rho.setValue((np.sqrt(2.))**(1./gamma), where=(x > 0.5) & (x < 0.7))

rho=Rho.value
U1=U.value


U_fig = CellVariable(name='$U$', mesh=mesh, value=0., hasOld=True)


# Rho1.setValue(np.exp(-(x-0.5)**2)+0.3)

def mu_rho(rho):
    N=len(rho)
    return 0.0001*rho


def Explicit_Staggered(rho, U1, dt, dx):
    N = len(rho)
    # equation de consevation de la masse
    diag0 = np.ones(N)  -(dt/dx) * (pplus(U1[1: N+1]) - pminus(U1[0: N]))
    diag0m= (dt/dx) * (pminus(U1[1:N+1]))
    diag0p= -(dt/dx) * (pplus(U1[0: N]))
    A = sp.lil_matrix(sp.spdiags([diag0m, diag0, diag0p], [-1, 0, 1], N, N))
    A[0, N-1] = dt/dx * (pplus(U1[0]))
    A[N-1, 0] = -dt/dx *(pminus(U1[N]))
    Rho_new= A.dot(rho)
    Rho_new_inter=(Rho_new+shiftd(Rho_new))/2.
    Rho_newF=np.concatenate([Rho_new_inter,[Rho_new_inter[0]]])

    # Convection: explicite
    rho_inter=(rho+shiftd(rho))/2.
    rhoF=np.concatenate([rho_inter,[rho_inter[0]]]) # rho sur les faces
    Uc=((U1+shiftg(U))/2.)[0:N] # U sur les mailles
    diag0_conv=rhoF*(np.concatenate([pplus(Uc)-pminus(shiftd(Uc)),[(pplus(Uc)-pminus(shiftd(Uc)))[0]]]))
    diag0p_conv=rhoF*np.concatenate([[0],pminus(Uc)])
    diag0m_conv=-rhoF*np.concatenate([pplus(Uc),[0]])
    A_conv = sp.lil_matrix(sp.spdiags([diag0m_conv, diag0_conv, diag0p_conv], [-1, 0, 1], N+1, N+1))
    A_conv[0,N]=-rhoF[0]*pplus(Uc[N-1])
    A_conv[N,0]=rhoF[0]*pminus(Uc[0])

    # Diffusion: implicite
    Mu_rho=mu_rho(Rho_new)
    coeff_mu_rho=np.concatenate([Mu_rho+shiftd(Mu_rho),[(Mu_rho+shiftd(Mu_rho))[0]]])
    diag0_diff=-(1/dx)*coeff_mu_rho
    diag0p_diff=(1/dx)*np.concatenate([[0],Mu_rho])
    diag0m_diff=(1/dx)*np.concatenate([Mu_rho,[0]])
    A_diff = sp.lil_matrix(sp.spdiags([diag0m_diff, diag0_diff, diag0p_diff], [-1, 0, 1], N+1, N+1))
    A_diff[0,N]=(1/dx)*Mu_rho[N-1]
    A_diff[N,0]=(1/dx)*Mu_rho[0]

    # Terme de pression
    P=barotrope_rhotoP(rho,c,gamma)
    P_inter=P-shiftd(P)
    P_dynamic=np.concatenate([P_inter,[P_inter[0]]])

    # Calcul de U en n+1
    Aright=sp.lil_matrix(sp.spdiags([(dx/dt)*Rho_newF], [0], N+1, N+1))+A_diff
    V_left=(dx/dt)*rhoF*U1-A_conv.dot(U1)-P_dynamic
    U_new=splin.gmres(Aright,V_left)[0]
    return Rho_new, U_new

# Boucle en temps
dt1 = 1e-4
duration = 1
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
    print('U_new: ',U_new)
    print('Rho_new: ', Rho_new)
    tps = tps + dt
    viewers.plot()
