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

def sol_rho(t,x):
    return 1+(1/4)*np.cos(np.pi*t)*np.sin(np.pi*x)

def sol_u(t,x):
    return (1/4)*np.cos(np.pi*t)*np.sin(np.pi*x)

def source(t,x):
    return -(1/16)*np.sin(np.pi*t)*np.pi*np.sin(np.pi*x)**2*np.cos(np.pi*t)-(1/4*(1+(1/4)*np.cos(np.pi*t)*np.sin(np.pi*x)))*np.sin(np.pi*t)*np.pi*np.sin(np.pi*x)+(1/64)*np.cos(np.pi*t)**3*np.cos(np.pi*x)*np.pi*np.sin(np.pi*x)**2\
           +((1/8)*(1+(1/4)*np.cos(np.pi*t)*np.sin(np.pi*x)))*np.cos(np.pi*t)**2*np.sin(np.pi*x)*np.cos(np.pi*x)*np.pi+(1/2*(1+(1/4)*np.cos(np.pi*t)*np.sin(np.pi*x)))*np.cos(np.pi*t)*np.cos(np.pi*x)*np.pi-0.625e-4*np.cos(np.pi*t)**2*np.cos(np.pi*x)**2*np.pi**2\
           +(0.25e-3*(1+(1/4)*np.cos(np.pi*t)*np.sin(np.pi*x)))*np.cos(np.pi*t)*np.sin(np.pi*x)*np.pi**2




