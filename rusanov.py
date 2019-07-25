from fipy import *
import numpy as np
from matplotlib import pyplot as plt
from fractions import Fraction
# plt.ion()
# plt.switch_backend('qt5agg')
#Maillage 1D
L = 1.0
nx = 10000
dx = L / nx

mesh = PeriodicGrid1D(dx, nx)
x, = mesh.cellCenters
# XF, =mesh.faceCenters
# Nvol=mesh.numberOfCells
# Nface=mesh.numberOfFaces

# ----------------------------------- A 2D Unstructured Mesh----------------------
# Lx = 40.0
# Ly = 80.
# nx = 40
# dx = Lx / nx
#
# cellSize = 1.5
# radius = Lx
# mesh = Gmsh2D('''
#         cellSize = %(cellSize)g;
#         radius   = %(radius)g;
#         Point(2) = {-radius, radius, 0, cellSize};
#         Point(3) = {2*radius, radius, 0, cellSize};
#         Point(4) = {2*radius, -radius, 0, cellSize};
#         Point(5) = {-radius, -radius, 0, cellSize};
#         Line(6) = {2, 3};
#         Line(7) = {3, 4};
#         Line(8) = {4, 5};
#         Line(9) = {5, 2};
#         Line Loop(10) = {6, 7, 8, 9};
#         Plane Surface(11) = {10};
#          ''' % locals())
# ---------------------------------- A 2D Structured Mesh ------------------------------------------

# cellSize = 0.5
# radius = Lx
# mesh = Gmsh2D('''
#         cellSize = %(cellSize)g;
#         meshThickness = cellSize / 10;
#         radius   = %(radius)g;
#         Point(2) = {-radius, radius, 0, cellSize};
#         Point(3) = {2*radius, radius, 0, cellSize};
#         Point(4) = {2*radius, -radius, 0, cellSize};
#         Point(5) = {-radius, -radius, 0, cellSize};
#         Line(6) = {2, 3};
#         Line(7) = {3, 4};
#         Line(8) = {4, 5};
#         Line(9) = {5, 2};
#         Line Loop(10) = {6, 7, 8, 9};
#         Plane Surface(11) = {10};
#         surfaceVector[] = Extrude {0, 0, meshThickness} {
#         Surface{11};
#         Layers{1};
#         Recombine;
#         };
#          ''' % locals())


# mesh = Grid2D(dx, dx, nx, nx, Lx=Lx, Ly=Ly)

# x, y = mesh.cellCenters

# -------------------------------------------------------------------------------------------------
nVol = mesh.numberOfCells
nFaces = mesh.numberOfFaces

# Faces
FacesIds = np.arange(0, nFaces, 1)
intFaces = mesh.interiorFaceIDs
extFaces = np.delete(numerix.arange(0, nFaces, 1), intFaces)
intFacesCells = mesh.faceCellIDs[:, intFaces]
extFacesCells = mesh.faceCellIDs[:, extFaces]
FacesCells = mesh.faceCellIDs
FacesCells[1][nFaces-1] = 0

# Cells
einT = np.arange(1, nVol-1, 1)
eexT = np.array([0, nVol-1])

CellFaces = mesh.cellFaceIDs
intCellFaces = mesh.cellFaceIDs[:,einT]
extCellFaces = mesh.cellFaceIDs[:,eexT]

# NS barotrope
c = 1.
gamma = 2.



# Variables
U1 = CellVariable(name='$u_1$', mesh=mesh, value=1., hasOld=True)

# U1.setValue(1, where=x >= 0.5)
# U1.setValue((np.sqrt(2))**(1./gamma), where=x < 0.5)
# test = 2.*np.ones(nVol)
# test[np.arange(1, nVol, 2)] = 1.
# U1.setValue(test)
U1.setValue(1., where=x >= 0.7)
U1.setValue(1., where=x < 0.5)
U1.setValue((np.sqrt(2.))**(1./gamma), where=(x > 0.5) & (x < 0.7))

# U1.setValue(np.exp(-((x-0.5)**2)/(2 * 0.005)))

# U1.setValue(np.exp(-((x-0.25)**2)/(2 * 0.005)) + np.exp(-((x-0.75)**2)/(2 * 0.005)))

U2 = CellVariable(name='$u_2$', mesh=mesh, value=0., hasOld=True)


Flux1 = FaceVariable(name="\mathcal{F_1}", mesh=mesh, value=0.)
Flux2 = FaceVariable(name="\mathcal{F_2}", mesh=mesh, value=0.)

dt1 = 1e-3
duration = 100
Nt = int(duration / dt1) + 1
dt = dt1


# Redefinition of numpy Power function
# def power_modified(x, y):
#     if np.mod(y, 2) == 0:
#         return (x)**(y)
#     return np.sign(x)*(np.abs(x))**(y)

# Flux
def Fe(x):
    # return np.array([x[1], (((x[1])**2)/x[0])+c*(x[0])**gamma])
    return x


def centered_mean(x,y):
    return (x+y)/2.


def harmonic_mean(x,y):
    return 2*(x*y)/(x + y)
# Loop in time

#shift
def shiftg(x):
    b=len(x)
    return np.concatenate([x[np.arange(1, b, 1)], np.array([x[0]])])

def shiftd(x):
    b=len(x)
    return x[np.arange(-1, b-1, 1)]

# Draw figures
# fig = plt.figure();
# fig.canvas.draw();


sp, axes = plt.subplots(1, 2)

Rho = Matplotlib1DViewer(vars=U1, axes=axes[0],
                         interpolation='spline16', figaspect='auto')

Rho_u_Rho = Matplotlib1DViewer(vars=U2/U1, axes=axes[1], datamin=0.000,
                               interpolation='spline16', figaspect='auto')

viewers = MultiViewer(viewers=(Rho, Rho_u_Rho))

tps = 0.
while tps <= duration:

    if U1.value.any() < 0.:
        break

    # The construction of Rusanov Fluxes
    U = np.array([U1, U2])
    Flux = np.array([Flux1, Flux2])
    # max_lambdas = np.zeros(nVol)
    max_lambdas = np.ones(nFaces)

    # --------------------------- The interior Faces ------------------
    lambda1 = np.abs((U2/U1) - numerix.sqrt(c*gamma*(U1)**(gamma-1)))
    lambda2 = np.abs((U2/U1) + numerix.sqrt(c*gamma*(U1)**(gamma-1)))

    max_lambdas = numerix.maximum(numerix.maximum(lambda1[FacesCells[0]], lambda1[FacesCells[1]]),
                                  numerix.maximum(lambda2[FacesCells[0]], lambda2[FacesCells[1]]))

    # print(max_lambdas)

    tem_Flux = (centered_mean(Fe(U[:, FacesCells[0]]), Fe(U[:, FacesCells[1]])) - (dx/dt) * (U[:, FacesCells[1]] - U[:, FacesCells[0]]) / 2.)
    # tem_Flux = (centered_mean(Fe(U[:, FacesCells[0]]), Fe(U[:, FacesCells[1]])) - max_lambdas * (U[:, FacesCells[1]] - U[:, FacesCells[0]]) / 2.)


    # shift ids to the left for the fluxes at i-0.5
    # Flux_moins = (centered_mean(Fe(U[:, shiftg(FacesCells[0])]), Fe(U[:, shiftg(FacesCells[1])]))) - max_lambdas * (U[:, shiftg(FacesCells[1])] - U[:, shiftg(FacesCells[0])]) / 2.

    # temp_Flux = Flux_plus - Flux_moins

    # Flux = Flux_plus[:, CellFaces[1]] - Flux_moins[:, CellFaces[0]]
    Flux[0] = tem_Flux[0] - shiftg(tem_Flux[0])
    Flux[1] = tem_Flux[1] - shiftg(tem_Flux[1])

    Unew = U - (dt/dx)*Flux[:, 1:nFaces]

    U1.setValue(Unew[0])
    U2.setValue(Unew[1])

    U1.updateOld()
    U2.updateOld()

    dt = np.min([dt1, 0.8 * dx / (np.max(max_lambdas.value))])

    tps = tps + dt

    print('time:{0}, dt: {1}'.format(tps, dt))
    # if np.isnan(dt):
    #     break

    viewers.plot()







