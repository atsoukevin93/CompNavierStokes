from fipy import *
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sp;
import scipy.sparse.linalg as splin;
from scipy.optimize import newton
from ToolBox import *
# import rusanov.Rusanov as Ru
from Diffusion1D import *

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
nVol = mesh.numberOfCells
nFaces = mesh.numberOfFaces


# Faces associées aux cellules (en périodique)
FacesCells = mesh.faceCellIDs


# Paramètre p(rho)=c*rho^{gamma}
c = 1.
gamma = 2.

def C_pressure(X,P,c,gamma,M, dx):
    return dx*np.sum(((P+X)/c)**(1/gamma))-M

def barotrope(P, c, gamma):
    return (P/c)**(1/gamma)


def Renormalization_step(P0, P1, c, gamma, L, dx):
    Rho0 = barotrope(P0, c, gamma)
    Rho1 = barotrope(P1, c, gamma)

    Diff1 = Build_Diffusion_Matrix(nVol, inv_vect(Phi_Rho(Rho1)), dx)
    Diff2 = Build_Diffusion_Matrix(nVol, inv_vect(Phi_Rho(Rho1)*Phi_Rho(Rho0))**0.5, dx)

    P_tild = splin.spsolve(Diff1.tocsc(), numerix.dot(Diff2, P1))
    M=dx*np.sum(Rho0)
    X=0
    Cnst=newton(C_pressure,X,args=[P_tild,c,gamma,M,dx],tol=0.00001,maxiter=50)
    return P_tild+Cnst

def pplus(x):
    return (x+np.abs(x))/2.
def pminus(x):
    return (x-np.abs(x))/2.

def mu_rho(rho):
    return rho


def convection_hkl(U,rho):
    N=len(rho)
    F=(U+shiftg(U))/2.
    rho1=np.concatenate([rho,[rho[0]]])
    rhoN=np.concatenate([[rho[N-1]],rho])
    diag0=rho1*pplus(F)+rhoN*pminus(shiftd(F))
    diag0p=np.concatenate([[0],-rho*(pminus(F)[0:N])])
    diag0m=np.concatenate([-rho*(pplus(F)[0:N]),[0]])
    M = sp.lil_matrix(sp.spdiags([diag0m, diag0, diag0p], [-1, 0, 1], N+1, N+1))
    M[0,N]=-rho[N-1]*pplus((U[0]+U[N])/2.)
    M[N,0]=-rho[0]*pminus((U[0]+U[N])/2.)
    return M

def diffusion_hkl(rho,dx):
    N=len(rho)
    F=mu_rho(rho)
    diag0=np.concatenate([F+shiftd(F),[(F+shiftd(F))[0]]])
    diag0p=np.concatenate([[0],-mu_rho(rho)])
    diag0m=np.concatenate([-mu_rho(rho),[0]])
    M = sp.lil_matrix(sp.spdiags([diag0m, diag0, diag0p], [-1, 0, 1], N + 1, N + 1))
    M[0,N]=-F[N-1]
    M[N,0]=-F[0]
    return (1/dx)*M

def linear_step3(P,U,rho,dx,dt):
    N = len(P)
    diag0=-(1/dx)*(P-shiftd(P))/(rho + shiftd(rho))+(1/dx)*(shiftg(P)-P)/(shiftg(rho)+rho)+(1/(2*dt))*((shiftg(U)-U)[0:N])
    diag0p= np.concatenate([[0],((1/dx)*(shiftg(P)-P)/(shiftg(rho)+rho)+(1/(2*dt))*shiftg(U)[0:N])[0:N-1]])
    diag0m=-(1/dx)*(shiftg(P)-P)/(shiftg(rho)+rho)-(1/(2*dt))*shiftg(U)[0:N]
    M = sp.lil_matrix(sp.spdiags([diag0m, diag0, diag0p], [-1, 0, 1], N, N))
    M[0, N-1] = diag0m[N-1]
    M[N-1,0] = ((1/dx)*(shiftg(P)-P)/(shiftg(rho)+rho)+(1/(2*dt))*shiftg(U)[0:N])[N-1]
    return  M


