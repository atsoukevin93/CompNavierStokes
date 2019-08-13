from fipy import *
import numpy as np
# from matplotlib import pyplot as plt
import scipy.sparse as sp;
import scipy.sparse.linalg as splin;
from ToolBox import *


def Build_Diffusion_Matrix(nVol, Phi_Rho, dx):
    if len(Phi_Rho)==0:
        raise "Not a Vector"
    else:
        Diff = sp.lil_matrix(sp.spdiags([Phi_Rho[0:nVol], -(Phi_Rho + shiftd(Phi_Rho)), shiftd(Phi_Rho[0:nVol])], [-1, 0, 1], nVol, nVol))
        Diff[0, nVol-1] = Phi_Rho[nVol-1]
        Diff[nVol-1, 0] = Phi_Rho[nVol-1]

        return (1/dx)*Diff
