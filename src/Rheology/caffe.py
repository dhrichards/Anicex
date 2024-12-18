import ufl
from ufl import i,j,k,l


def Deformability(a2, a4, ε):
    I = ufl.Identity(a2.ufl_shape[0])
    return 5*(ε[i,j]*ε[k,l]*I[i,k]*a2[j,l] - ε[i,j]*ε[k,l]*a4[i,j,k,l])/ufl.inner(ε,ε)



def Enhancement(D,Emin=0.1,Emax=10.0):
    τ = (8/21)*((Emax-1)/(1-Emin))

    Elow = (1-Emin)*D**τ + Emin
    Ehigh = (4*D**2*(Emax-1) + 25 - 4*Emax)/21

    return ufl.conditional(ufl.ge(D,1.0),Ehigh,Elow)
