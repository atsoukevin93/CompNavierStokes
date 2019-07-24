from fipy import *
import numpy as np
from matplotlib import pyplot as plt

#Maillage 1D
L = 1.0
nx = 100
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
U1.setValue(1, where=x < 0.5)
U1.setValue((np.sqrt(2))**(1./gamma), where=x >= 0.5)
# U1.setValue(np.exp(-(x-0.5)**2))

U2 = CellVariable(name='$u_2$', mesh=mesh, value=0., hasOld=True)

Flux1 = CellVariable(name="\mathcal{F_1}", mesh=mesh, value=0.)
Flux2 = CellVariable(name="\mathcal{F_2}", mesh=mesh, value=0.)

dt = 0.01
duration = 10
Nt = int(duration / dt) + 1


# Flux
def Fe(x):
    return np.array([x[1],((x[1]**2)/x[0])+c*x[0]**gamma])

# Loop in time


# Draw figures
# fig = plt.figure();
# fig.canvas.draw();
sp, axes = plt.subplots(1, 2)

Rho = Matplotlib1DViewer(vars=U1, axes=axes[0],
                         interpolation='spline16', figaspect='auto')

Rho_u_Rho = Matplotlib1DViewer(vars=U2/U1, axes=axes[1], datamin=0.000,
                               interpolation='spline16', figaspect='auto')

viewers = MultiViewer(viewers=(Rho, Rho_u_Rho))

for n in range(Nt):
    # The construction of the Fluxes
    U = np.array([U1, U2])
    Flux = np.array([Flux1, Flux2])
    lambdas = np.zeros(nVol)

    # --------------------------- The interior Faces ------------------
    lambda1 = ((U[1]/U[0])- numerix.sqrt(c*gamma*U[0]**(gamma-1)))
    lambda2 = ((U[1]/U[0]) + numerix.sqrt(c*gamma*U[0]**(gamma-1)))

    lambdas = numerix.maximum(numerix.maximum(lambda1[FacesCells[0]], lambda1[FacesCells[1]]),
                              numerix.maximum(lambda2[FacesCells[0]], lambda2[FacesCells[1]]))

    Flux_plus = (Fe(U[:, FacesCells[0]]) + Fe(U[:, FacesCells[1]]))/2. - lambdas * (U[:, FacesCells[1]] - U[:, FacesCells[0]])/2.

    # shift ids to the left for the fluxes at i-0.5
    Flux_moins = (Fe(U[:, FacesCells[0]-1]) + Fe(U[:, FacesCells[1]-1]))/2. - lambdas * (U[:, FacesCells[1]-1] - U[:, FacesCells[0]-1])/2.

    temp_Flux = Flux_plus - Flux_moins

    Flux[:, :] = Flux_plus[:, CellFaces[1]] - Flux_moins[:, CellFaces[0]]

    dt = dx/np.max(lambdas)

    print(dt)

    Unew = U - (dt/dx)*Flux

    U1.setValue(Unew[0])
    U2.setValue(Unew[1])

    U1.updateOld()
    U2.updateOld()

    viewers.plot()







