from fipy import *
import numpy as np
from matplotlib import pyplot as plt

#Maillage 1D
L = 1.0
nx = 100
dx = L / nx

mesh = Grid1D(dx, nx)
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
intFaces = mesh.interiorFaceIDs
extFaces = np.delete(numerix.arange(0, nFaces, 1), intFaces)
intFacesCells = mesh.faceCellIDs[:, intFaces]
extFacesCells = mesh.faceCellIDs[:, extFaces]
FaceNormals = mesh.faceNormals
FaceNormalsext = FaceNormals[:, extFaces]
FaceNormalsint = FaceNormals[:, intFaces]
LeftFacesIds = numerix.where(np.logical_and(FaceNormalsext[0, ] == -1., FaceNormalsext[1, ] == 0.))[0]
RightFacesIds = numerix.where(np.logical_and(FaceNormalsext[0, ] == 1., FaceNormalsext[1, ] == 0.))[0]
DownFacesIds = numerix.where(np.logical_and(FaceNormalsext[0, ] == 0., FaceNormalsext[1, ] == -1.))[0]
TopFacesIds = numerix.where(np.logical_and(FaceNormalsext[0, ] == 0., FaceNormalsext[1, ] == 1.))[0]

mesF = mesh._faceAreas

# Cells
einT = mesh._interiorCellIDs[0]
eexT = mesh._exteriorCellIDs[0]
intCellFaces = mesh.cellFaceIDs[:, einT]
extCellFaces = mesh.cellFaceIDs[:, eexT]
#Constantes

#NS barotrope
c=1
gamma=2



#Variables
u = CellVariable(name='$u$', mesh=mesh, rank=1, hasOld=True)
# u.setValue(np.ones(nVol))
u[0] = np.ones(nVol)
u[1] = np.ones(nVol)

Rho = CellVariable(name='$\\rho$', mesh=mesh, value=0, hasOld=True)
Rs = 1
# Rho.setValue(numerix.exp(-((x)**2)/(2*Rs))/(numerix.sqrt(2*numerix.pi*Rs)))
# Rho.setValue(numerix.exp(-((x-40.)**2 + (y)**2)/(2*Rs)))
Rho.setValue(numerix.exp(-((x-40.)**2 + (y)**2)/(2*Rs)))

Rho_u = CellVariable(name='$\\rho u $', mesh=mesh, value=0., rank=1, hasOld=True)
Rho_u.setValue(Rho*u)

Flux1 = FaceVariable(name="\mathcal{F}_1", mesh=mesh, value=0., rank=1)
Flux2 = FaceVariable(name="\mathcal{F}_2", mesh=mesh, value=0., rank=1)
# temp_int_Flux = FaceVariable(name="\mathcal{F}_{in}", mesh=mesh, value=0., rank=1)
# temp_ext_Flux = FaceVariable(name="\mathcal{F}_{ext}", mesh=mesh, value=0., rank=1)
Flux = CellVariable(name="\mathcal{F}", mesh=mesh, value=0., rank=1)

dt = 0.01
duration = 120
Nt = int(duration / dt) + 1


# first component of the flux
def F_1(x, y):
    return x*y


# second component of the flux
def F_2(x, y):
    return x*(y**2) + c*(y**gamma)

# Loop in time


for n in range(Nt):
    # The construction of the Fluxes
    temp_Flux = np.zeros((2, 2, nFaces))
    temp_FluxNormals = np.zeros(( 2, nFaces))
    # --------------------------- The interior Faces ------------------
    lambda1 = 2*(u + numerix.sqrt(c*gamma*Rho**(gamma-1)))
    lambda2 = 2*(u - numerix.sqrt(c*gamma*Rho**(gamma-1)))
    lambdas_int = numerix.maximum(numerix.maximum(lambda1[:,intFacesCells[0]],lambda1[:,intFacesCells[1]]),
                                  numerix.maximum(lambda2[:,intFacesCells[0]],lambda2[:,intFacesCells[1]]))

    Flux1[:,intFaces] = (((F_1(Rho[intFacesCells[0]], u[:,intFacesCells[0]]) + F_1(Rho[intFacesCells[1]], u[:,intFacesCells[1]]))/2.)
                   - lambdas_int * (Rho[intFacesCells[1]] - Rho[intFacesCells[0]])/2.)
    Flux2[:,intFaces] = (((F_2(Rho[intFacesCells[0]], u[:,intFacesCells[0]]) + F_2(Rho[intFacesCells[1]], u[:,intFacesCells[1]]))/2.)
                   - lambdas_int * (Rho[intFacesCells[1]] - Rho[intFacesCells[0]])/2.)

    temp_Flux[0][:, intFaces] = Flux1[:, intFaces]
    temp_Flux[1][:, intFaces] = Flux2[:, intFaces]
    temp_int_Flux = np.array([temp_Flux[0][:, intFaces],temp_Flux[1][:, intFaces]])
    temp_int_FluxNormals = mesF[intFaces] * numerix.dot(temp_int_Flux, mesh.faceNormals[:, intFaces])
    temp_FluxNormals[:,intFaces] = temp_int_FluxNormals

    # The sum of the fluxes on the faces of the interior control
    # np.where()
    Flux[:,einT] = temp_FluxNormals[:,intCellFaces[0]] + temp_FluxNormals[:,intCellFaces[1]] + temp_FluxNormals[:,intCellFaces[2]]

    # --------------------------- The exterior Faces  with periodic Boundary Conditions------------------
    lambdas_ext_left = numerix.maximum(numerix.maximum(lambda1[:,extFacesCells[1, RightFacesIds]],
                                       lambda1[:,extFacesCells[0, LeftFacesIds]]),
                                       numerix.maximum(lambda2[:,extFacesCells[1, RightFacesIds]],
                                       lambda2[:,extFacesCells[0, LeftFacesIds]]))

    lambdas_ext_right = numerix.maximum(numerix.maximum(lambda1[:,extFacesCells[0, RightFacesIds]],
                                       lambda1[:,extFacesCells[1, LeftFacesIds]]),
                                        numerix.maximum(lambda2[:,extFacesCells[0, RightFacesIds]],
                                       lambda2[:,extFacesCells[1, LeftFacesIds]]))

    lambdas_ext_top = numerix.maximum(numerix.maximum(lambda1[:,extFacesCells[0, TopFacesIds]],
                                       lambda1[:,extFacesCells[1, DownFacesIds]]),
                                      numerix.maximum(lambda2[:,extFacesCells[0, TopFacesIds]],
                                       lambda2[:,extFacesCells[1, DownFacesIds]]))

    lambdas_ext_down = numerix.maximum(numerix.maximum(lambda1[:,extFacesCells[0, DownFacesIds]],
                                       lambda1[:,extFacesCells[1, TopFacesIds]]),
                                       numerix.maximum(lambda2[:,extFacesCells[0, DownFacesIds]],
                                       lambda2[:,extFacesCells[1, TopFacesIds]]))

    Flux1[:,LeftFacesIds] = (((F_1(Rho[extFacesCells[0, RightFacesIds]], u[:,extFacesCells[0, RightFacesIds]]) + F_1(Rho[extFacesCells[1, LeftFacesIds]], u[:,extFacesCells[1, LeftFacesIds]]))/2.)
                   - lambdas_ext_left * (Rho[extFacesCells[1, LeftFacesIds]] - Rho[extFacesCells[0, RightFacesIds]])/2.)
    Flux1[:,RightFacesIds] = (((F_1(Rho[extFacesCells[0, LeftFacesIds]], u[:,extFacesCells[0, LeftFacesIds]]) + F_1(Rho[extFacesCells[1, RightFacesIds]], u[:,extFacesCells[1, RightFacesIds]]))/2.)
                   - lambdas_ext_right * (Rho[extFacesCells[1, RightFacesIds]] - Rho[extFacesCells[0, LeftFacesIds]])/2.)
    Flux1[:,TopFacesIds] = (((F_1(Rho[extFacesCells[0, TopFacesIds]], u[:,extFacesCells[0, TopFacesIds]]) + F_1(Rho[extFacesCells[1, DownFacesIds]], u[:,extFacesCells[1, DownFacesIds]]))/2.)
                   - lambdas_ext_top * (Rho[extFacesCells[1, DownFacesIds]] - Rho[extFacesCells[0, TopFacesIds]])/2.)
    Flux1[:,TopFacesIds] = (((F_1(Rho[extFacesCells[0, TopFacesIds]], u[:,extFacesCells[0, TopFacesIds]]) + F_1(Rho[extFacesCells[1, DownFacesIds]], u[:,extFacesCells[1, DownFacesIds]]))/2.)
                   - lambdas_ext_down * (Rho[extFacesCells[1, DownFacesIds]] - Rho[extFacesCells[0, TopFacesIds]])/2.)

    Flux2[:,LeftFacesIds] = (((F_2(Rho[extFacesCells[0, RightFacesIds]], u[:,extFacesCells[0, RightFacesIds]]) + F_2(
        Rho[extFacesCells[1, LeftFacesIds]], u[:,extFacesCells[1, LeftFacesIds]])) / 2.)
                                 - lambdas_ext_left * (
                                 Rho[extFacesCells[1, LeftFacesIds]] - Rho[extFacesCells[0, RightFacesIds]]) / 2.)
    Flux2[:,RightFacesIds] = (((F_2(Rho[extFacesCells[0, LeftFacesIds]], u[:,extFacesCells[0, LeftFacesIds]]) + F_2(
        Rho[extFacesCells[1, RightFacesIds]], u[:,extFacesCells[1, RightFacesIds]])) / 2.)
                                  - lambdas_ext_right * (
                                  Rho[extFacesCells[1, RightFacesIds]] - Rho[extFacesCells[0, LeftFacesIds]]) / 2.)
    Flux2[:,TopFacesIds] = (((F_2(Rho[extFacesCells[0, TopFacesIds]], u[:,extFacesCells[0, TopFacesIds]]) + F_2(
        Rho[extFacesCells[1, DownFacesIds]], u[:,extFacesCells[1, DownFacesIds]])) / 2.)
                                - lambdas_ext_top * (
                                Rho[extFacesCells[1, DownFacesIds]] - Rho[extFacesCells[0, TopFacesIds]]) / 2.)
    Flux2[:,TopFacesIds] = (((F_2(Rho[extFacesCells[0, TopFacesIds]], u[:,extFacesCells[0, TopFacesIds]]) + F_2(
        Rho[extFacesCells[1, DownFacesIds]], u[:,extFacesCells[1, DownFacesIds]])) / 2.)
                                - lambdas_ext_down * (
                                Rho[extFacesCells[1, DownFacesIds]] - Rho[extFacesCells[0, TopFacesIds]]) / 2.)

    temp_Flux[0][:, extFaces] = Flux1[:, extFaces]
    temp_Flux[1][:, extFaces] = Flux2[:, extFaces]
    temp_ext_Flux = np.array([temp_Flux[0][:, extFaces], temp_Flux[1][:, extFaces]])
    temp_ext_FluxNormals = mesF[extFaces] * numerix.dot(temp_ext_Flux, mesh.faceNormals[:, extFaces])
    temp_FluxNormals[:, extFaces] = temp_ext_FluxNormals

    # temp_ext_Flux = np.array([Flux1.value[:, extFaces], Flux2.value[:, extFaces]])
    # temp_ext_FluxNormals = mesF[extFaces] * numerix.dot(temp_ext_Flux, mesh.faceNormals[:, extFaces])

    # The sum of the fluxes on the faces of the interior control
    Flux[:,eexT] = numerix.sum(temp_ext_FluxNormals[extCellFaces], axis=0)

    Unew = np.array([Rho, Rho_u]) - (dt/dx)*Flux

    Rho.setValue(Unew[0])
    Rho_u.setValue(Unew[1])
    u.setValue(Rho_u/Rho)
    Rho.updateOld()
    Rho_u.updateOld()
    u.updateOld()

#Flux
# def Fe(x):
#     return np.array([x[1],((x[0]**2)/x[0])+c*x[0]**gamma])
#
# #Valeurs propres
# def eig(x):
#     return np.array([(x[1]/x[0])-np.sqrt(gamma*c*x[0]**(gamma-1)),(x[1]/x[0])+np.sqrt(gamma*c*x[0]**(gamma-1))])
#


#Calcul du flux
#def Flux(x,y):
 #   return ((Fe(y)-Fe(x))/2)-max()



