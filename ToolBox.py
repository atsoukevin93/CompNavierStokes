import numpy as np


def centered_mean(x,y):
    return (x+y)/2.


def shiftg(x):
    b=len(x)
    return np.concatenate([x[np.arange(1, b, 1)], np.array([x[0]])])


def shiftd(x):
    b=len(x)
    return x[np.arange(-1, b-1, 1)]


def pplus(x):
    return (x+np.abs(x))/2.


def pminus(x):
    return (x-np.abs(x))/2.


def barotrope_Ptorho(P, c, gamma):
    return (P/c)**(1/gamma)


def barotrope_rhotoP(rho, c, gamma):
    return c*rho**gamma


def Phi_Rho(mu_Rho):
    if len(mu_Rho)==0:
        raise "Not a Vector"
    else:
        return (shiftg(mu_Rho) + mu_Rho)/2.


def inv_vect(vect):
    if len(vect)==0:
        raise "Not a Vector"
    else:
        return 1/vect

