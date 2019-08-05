from fipy import *
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sp;
import scipy.sparse.linalg as splin;
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


def barotrope(P, c, gamma):
    return (P/c)**(1/gamma)


def Renormalization_step(P0, P1, c, gamma):
    Rho0 = barotrope(P0, c, gamma)
    Rho1 = barotrope(P1, c, gamma)

    Diff1 = Build_Diffusion_Matrix(nVol, inv_vect(Phi_Rho(Rho1)), dx)
    Diff2 = Build_Diffusion_Matrix(nVol, inv_vect(Phi_Rho(Rho1)*Phi_Rho(Rho0))**0.5, dx)

    P_tild = splin.spsolve(Diff1.tocsc(), numerix.dot(Diff2, P1))

    return P_tild
