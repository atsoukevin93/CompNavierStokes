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
nx = 50
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


# Variables
U1 = CellVariable(name='$u_1$', mesh=mesh, value=1., hasOld=True) #correspond à rho
U2 = CellVariable(name='$u_2$', mesh=mesh, value=0., hasOld=True) #correspond à rho*u, donnée initiale de la vitesse nulle
Flux1 = FaceVariable(name="\mathcal{F_1}", mesh=mesh, value=0.)
Flux2 = FaceVariable(name="\mathcal{F_2}", mesh=mesh, value=0.)


# Donnée initiale sur rho
U1.setValue(1., where=x >= 0.7)
U1.setValue(1., where=x < 0.5)
U1.setValue((np.sqrt(2.))**(1./gamma), where=(x > 0.5) & (x < 0.7))



# Paramètres graphiques
sp1, axes = plt.subplots(1, 2)

Rho = Matplotlib1DViewer(vars=U1, axes=axes[0], interpolation='spline16', figaspect='auto')

Rho_u_Rho = Matplotlib1DViewer(vars=U2/U1, axes=axes[1], interpolation='spline16', figaspect='auto')

viewers = MultiViewer(viewers=(Rho, Rho_u_Rho))


def Fe(x):
    return np.array([x[1], (((x[1]) ** 2) / x[0]) + c * (x[0]) ** gamma])


def Rusanov(U, dt, dx):
    Flux1 = FaceVariable(name="\mathcal{F_1}", mesh=mesh, value=0.)
    Flux2 = FaceVariable(name="\mathcal{F_2}", mesh=mesh, value=0.)

    Flux = np.array([Flux1, Flux2])

    # Calcul du max des valeurs propres
    lambda1 = np.abs((U2 / U1) - numerix.sqrt(c * gamma * (U1) ** (gamma - 1)))
    lambda2 = np.abs((U2 / U1) + numerix.sqrt(c * gamma * (U1) ** (gamma - 1)))

    # Correction de Rusanov sur chaque faces
    max_lambdas = numerix.maximum(numerix.maximum(lambda1[FacesCells[0]], lambda1[FacesCells[1]]),
                                  numerix.maximum(lambda2[FacesCells[0]], lambda2[FacesCells[1]]))

    # Calcul du flux de Rusanov sur chaque faces
    tem_Flux = (centered_mean(Fe(U[:, FacesCells[0]]), Fe(U[:, FacesCells[1]])) - max_lambdas * (
    U[:, FacesCells[1]] - U[:, FacesCells[0]]) / 2.)  # Rusanov
    # tem_Flux = (centered_mean(Fe(U[:, FacesCells[0]]), Fe(U[:, FacesCells[1]])) - (dx/dt) * (U[:, FacesCells[1]] - U[:, FacesCells[0]]) / 2.) #Lax-Friedrichs


    # Calcul du flux global
    Flux[0] = shiftg(tem_Flux[0]) - tem_Flux[0]
    Flux[1] = shiftg(tem_Flux[1]) - tem_Flux[1]

    Unew = U - (dt/dx)*Flux[:, 0:nFaces-1] #Mise à jour de U, Flux=[f0-f-1, f1-f0,....,f-1-f-2,f0-f-1], on coupe le dernier morceau

    U1.setValue(Unew[0])
    U2.setValue(Unew[1])

    return np.array([U1, U2])

def Phi_Rho(mu_Rho):
    if len(mu_Rho)==0:
        raise "Not a Vector"
    else:
        return (shiftg(mu_Rho) + mu_Rho)/2.


def mu_Rho(x,c):
    return c*x

# Boucle en temps
dt1 = 1e-3
duration = 100
Nt = int(duration / dt1) + 1
dt = dt1
tps = 0.

Id = sp.lil_matrix(sp.spdiags(numerix.ones(nVol), [0], nVol, nVol))

while tps <= duration:

    if U1.value.any() < 0.:
        break
    U = np.array([U1, U2])

    # Les valeurs propres
    lambda1 = np.abs((U2 / U1) - numerix.sqrt(c * gamma * (U1) ** (gamma - 1)))
    lambda2 = np.abs((U2 / U1) + numerix.sqrt(c * gamma * (U1) ** (gamma - 1)))

    # Correction de Rusanov sur chaque faces
    max_lambdas = numerix.maximum(numerix.maximum(lambda1[FacesCells[0]], lambda1[FacesCells[1]]),
                                  numerix.maximum(lambda2[FacesCells[0]], lambda2[FacesCells[1]]))

    # Etape Rusanov
    Ustar = Rusanov(U, dt, dx)
    # Ustar=U
    mu_Rho_star = mu_Rho(Ustar[0], 100.)
    phi_Rho_star = Phi_Rho(mu_Rho_star)

    # Matrice de Diffusion
    Diff = Build_Diffusion_Matrix(nVol, phi_Rho_star, dx)

    # Calcul du vecteur des vitesses dans la deuxieme etape du splitting
    U_ustar_new = splin.spsolve(Id - (dt/dx)*Diff, Ustar[1]/Ustar[0])

    # U_ustar_new = U
    U1.setValue(Ustar[0])
    U2.setValue(Ustar[0]*U_ustar_new)



    # Condition CFL
    dt = np.min([dt1, 0.8 * dx / (np.max(max_lambdas.value))])

    # Mise à jour du temps
    tps = tps + dt

    print('time:{0}, dt: {1}'.format(tps, dt))
    # if np.isnan(dt):
    #     break

    viewers.plot()
