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
import os


# Paramètre du maillage 1D
L = 1.0
nx =500
dx = L / nx


# Construction du maillage
mesh = PeriodicGrid1D(dx, nx)


# Données du maillage
# Centres des mailles
x, = mesh.cellCenters


# Nombre de volumes, nombre de faces
Nvol = mesh.numberOfCells
nFaces = mesh.numberOfFaces


# Faces associées aux cellules (en périodique)
FacesCells = mesh.faceCellIDs


# Paramètre p(rho)=c*rho^{gamma}
c = 1.
gamma = 3.

# les inconnus
U = FaceVariable(name='$u$', mesh=mesh, value=0.)
Rho0 = CellVariable(name='$\\rho$', mesh=mesh, value=0., hasOld=True)
Rho1 = CellVariable(name='$\\rho$', mesh=mesh, value=0., hasOld=True)
Rho2 = CellVariable(name='$\\rho$', mesh=mesh, value=0., hasOld=True)

# Donnée initiale sur rho
Rho0.setValue(1., where=x >= 0.7)
Rho0.setValue(1., where=x < 0.5)
Rho0.setValue((np.sqrt(2.))**(1./gamma), where=(x > 0.5) & (x < 0.7))

Rho1.setValue(1., where=x >= 0.7)
Rho1.setValue(1., where=x < 0.5)
Rho1.setValue((np.sqrt(2.))**(1./gamma), where=(x > 0.5) & (x < 0.7))

#
rho0=Rho0.value
rho1=Rho1.value
U1=U.value

U_fig = CellVariable(name='$U$', mesh=mesh, value=0., hasOld=True)


# Rho1.setValue(np.exp(-(x-0.5)**2)+0.3)


def C_pressure(X,P,c,gamma,M, dx):
    return dx*np.sum(((P+X)/c)**(1/gamma))-M


def Renormalization_step(P0, P1, c, gamma, L, dx, tol, maxiter):
    N=len(P0)
    Rho0 = barotrope_Ptorho(P0, c, gamma)
    Rho1 = barotrope_Ptorho(P1, c, gamma)

    Diff1 = Build_Diffusion_Matrix(N, inv_vect(Phi_Rho(Rho1)), dx)
    # print('Diff1 :',Diff1.todense())
    Diff2 = Build_Diffusion_Matrix(N, inv_vect(Phi_Rho(Rho1) * Phi_Rho(Rho0)) ** 0.5, dx)
    # print('Diff2 :', Diff2.todense())
    P_tild = splin.gmres(Diff1.tocsc(), numerix.dot(Diff2, P1))[0]
    M=dx*np.sum(Rho0)
    X_cop=c*(M/L)**(gamma)
    # print('M: ',M)
    # print('Ptild: ', P_tild)
    # print('dx*np.sum(P_tild): ',dx*np.sum(P_tild))
    Cnst=newton(C_pressure,X_cop,args=[P_tild,c,gamma,M,dx],tol=tol,maxiter=maxiter)
    # Cnst=(c/L)*M-(dx/L)*np.sum(P_tild)
    return P_tild+Cnst


def mu_rho(rho):
    N=len(rho)
    return 0.01*rho
    # return np.ones(N)


def convection_hkl(U,rho):
    N=len(rho)
    F=(U+shiftg(U))/2.
    rho1=np.concatenate([rho,[rho[0]]])
    rhoN=np.concatenate([[rho[N-1]],rho])
    diag0=rho1*pplus(F)-rhoN*pminus(shiftd(F))
    diag0p=np.concatenate([[0],rho*(pminus(F)[0:N])])
    diag0m=np.concatenate([-rho*(pplus(F)[0:N]),[0]])
    M = sp.lil_matrix(sp.spdiags([diag0m, diag0, diag0p], [-1, 0, 1], N+1, N+1))
    M[0,N]=-rho[N-1]*pplus((U[0]+U[N])/2.)
    M[N,0]=+rho[0]*pminus((U[0]+U[N])/2.)
    return M

def diffusion_hkl(rho,dx):
    N=len(rho)
    F=mu_rho(rho)
    diag0=np.concatenate([F+shiftd(F),[(F+shiftd(F))[0]]])
    diag0p=np.concatenate([[0],-mu_rho(rho)])
    diag0m=np.concatenate([-mu_rho(rho),[0]])
    M = sp.lil_matrix(sp.spdiags([diag0m, diag0, diag0p], [-1, 0, 1], N + 1, N + 1))
    M[0,N]=-F[N-1]
    M[N,0]=-F[0]
    return (1/dx)*M

def linear_step3(P,U,rho,dx,dt):
    N = len(P)
    diag0=(dx/(dt*dt))-(1/dx)*(P-shiftd(P))/(rho + shiftd(rho))+(1/dx)*(shiftg(P)-P)/(shiftg(rho)+rho)+(1/(2*dt))*(shiftg(U)-U)[0:N]
    diag0p= np.concatenate([[0],((1/dx)*(shiftg(P)-P)/(shiftg(rho)+rho)+(1/(2*dt))*shiftg(U)[0:N])[0:N-1]])
    diag0m=-(1/dx)*(shiftg(P)-P)/(shiftg(rho)+rho)-(1/(2*dt))*shiftg(U)[0:N]
    M = sp.lil_matrix(sp.spdiags([diag0m, diag0, diag0p], [-1, 0, 1], N, N))
    M[0, N-1] = -(1/dx)*(P[0]-P[N-1])/(rho[0]+rho[N-1])-(1/(2*dt))*U[0]
    M[N-1,0] = +(1/dx)*(P[0]-P[N-1])/(rho[0]+rho[N-1])+(1/(2*dt))*U[N]
    return  M

def nonlinear_step3(X,rho,dx):
    N=len(rho)
    Y=barotrope_rhotoP(X, c, gamma)
    diag0=-(1/dx)*(shiftg(Y)-Y)/(shiftg(rho)+rho)+(1/dx)*(Y-shiftd(Y))/(shiftd(rho)+rho)
    diag0p=np.concatenate([[0],(-(1/dx)*(shiftg(Y)-Y)/(shiftg(rho)+rho))[0:N-1]])
    diag0m=np.concatenate([((1/dx)*(Y-shiftd(Y))/(shiftd(rho)+rho))[1:N],[0]])
    M = sp.lil_matrix(sp.spdiags([diag0m, diag0, diag0p], [-1, 0, 1], N, N))
    M[0,N-1]=(1/dx)*(Y[0]-Y[N-1])/(rho[0]+rho[N-1])
    M[N-1,0]=-(1/dx)*(Y[0]-Y[N-1])/(rho[0]+rho[N-1])
    return M*X

def total_step3(X,P,U,rho,dx,dt):
    return (linear_step3(P,U,rho,dx,dt)*X)+nonlinear_step3(X,rho,dx)-(dx/(dt*dt))*rho

def step3(U,P,rho,dx,dt,tol,maxiter):
    rho_cop=np.copy(rho)
    return newton(total_step3,rho_cop,args=(P,U,rho,dx,dt),maxiter=maxiter,tol=tol)
    # return fsolve(total_step3,rho_cop,args=(P,U,rho,dx,dt),xtol=tol)


def step4(U,rho,P0,P1,dx,dt):
    X=((2*dt)/dx)*(shiftd(P1)-P1+P0-shiftd(P0))/(shiftd(rho)+rho)
    return U+np.concatenate([X, [X[0]]])

def HKL(rho0,rho1,U1,dx,dt,L,c,gamma,tol,maxiter):
    N=len(rho0)
    P0=barotrope_rhotoP(rho0,c,gamma)
    P1=barotrope_rhotoP(rho1,c,gamma)
    # Step 1
    Ptilde = Renormalization_step(P0,P1,c,gamma,L,dx, tol, maxiter)
    # Ptilde=P1
    rhotilde=barotrope_Ptorho(Ptilde,c,gamma)
    # Step 2
    Diag0=(dx/(2*dt))*np.concatenate([rhotilde+shiftd(rhotilde),[(rhotilde+shiftd(rhotilde))[0]]])
    D=sp.lil_matrix(sp.spdiags([Diag0], [0], N+1, N+1))
    B=D+convection_hkl(U1,rhotilde)+diffusion_hkl(rhotilde,dx)
    Diag1=(dx/(2*dt))*np.concatenate([rho1+shiftd(rho1),[(rho1+shiftd(rho1))[0]]])
    Y=Diag1*U1+np.concatenate([shiftd(Ptilde)-Ptilde,[(shiftd(Ptilde)-Ptilde)[0]]])
    Utilde=splin.gmres(B,Y)[0]
    # Step 3P
    rho2=step3(Utilde,Ptilde,rho1,dx,dt,tol,maxiter)
    P2=barotrope_rhotoP(rho2,c,gamma)
    # Step 4
    U2=step4(Utilde,rho2,Ptilde,P1,dx,dt)
    return [rho1, rho2, U2]

# figures
# sp1, axes = plt.subplots(1,2)
#
# Rho_fig = Matplotlib1DViewer(vars=Rho1, axes=axes[0], interpolation='spline16', datamax = 1.5, figaspect='auto')
#
# u_fig = Matplotlib1DViewer(vars=U_fig, axes=axes[1], interpolation='spline16', figaspect='auto')
#
# viewers = MultiViewer(viewers=(Rho_fig, u_fig))
# viewers = MultiViewer(viewers=(Rho_fig))

# Boucle en temps
dt1 = 1e-4
duration = 0.5
Nt = int(duration / dt1) + 1
dt = dt1
tps = 0.

test_case_results = np.empty([], dtype=[('t', np.float64),
                                        ('dt', np.float64),
                                        ('dx', np.float64),
                                        ('Rho', np.float64, (Nvol,)),
                                        ('U', np.float64, (nFaces,))])
test_case_results = np.delete(test_case_results, 0)
n=0
while tps <= duration:

    tol = 1e-8
    maxiter = 2000
    rho1, rho2, U2 = HKL(Rho0.value, Rho1.value, U.value, dx, dt, L, c, gamma, tol, maxiter)

    Rho0.setValue(rho1)
    Rho1.setValue(rho2)
    U.setValue(U2)
    U_fig.setValue((U + shiftg(U))[0:Nvol]/2.)

    if n % 2 == 0:
        test_case_results = np.append(test_case_results, np.asarray((tps, dt, dx, Rho1, U), dtype=test_case_results.dtype))
    # raw_input("pause...")
    # print(Rho1)
    # print('vitesse',U)
    tps = tps + dt
    print(tps)
    # time.sleep(10)
    n = n + 1
    # viewers.plot()

dirpath = "data/"
filename = "hkl_test"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

with open(dirpath+filename, "wb") as fi:
    np.save(fi, test_case_results)
    # np.save(fi, TestCaseParam)
    fi.close()
