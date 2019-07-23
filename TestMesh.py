#!/usr/bin/env python
# -*- coding: utf-8 -*-
from fipy import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

# ----------------------------------- A one dimensional Grid----------------------
L = 1.0
nx = 100
dx = L / nx

mesh = Grid1D(dx, nx)
x, = mesh.cellCenters

# ----------------------------------- A two dimensional Mesh----------------------
# cellSize = 0.5
# radius = L
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
#
# x, y = mesh.cellCenters
# nVol = mesh.numberOfCells

#mesh = Gmsh2D("/user/katsou/home/PycharmProjects/CompNavierStokes/GenMeshes/test2.msh")