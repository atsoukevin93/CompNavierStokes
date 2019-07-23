from fipy import *
import numpy as np


#Maillage 1D
L = 1.0
nx = 100
dx = L / nx

mesh = Grid1D(dx, nx)
X, = mesh.cellCenters
XF, =mesh.faceCenters
Nvol=mesh.numberOfCells
Nface=mesh.numberOfFaces

#Constantes

#NS barotrope
c=1
gamma=2



#Variables
U=np.ones(Nvol)


#Flux
def Fe(x):
    return np.array([x[1],((x[0]**2)/x[0])+c*x[0]**gamma])

#Valeurs propres
def eig(x):
    return np.array([(x[1]/x[0])-np.sqrt(gamma*c*x[0]**(gamma-1)),(x[1]/x[0])+np.sqrt(gamma*c*x[0]**(gamma-1))])



#Calcul du flux
#def Flux(x,y):
 #   return ((Fe(y)-Fe(x))/2)-max()



