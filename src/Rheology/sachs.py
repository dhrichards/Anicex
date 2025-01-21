import ufl
from Rheology import mandel
from ufl import i,j,k,l


def με(ε,a2,a4,Ecc,Eca):
    μ = sachs_visc(a2,a4,Ecc,Eca)
    return ufl.as_tensor(μ[i,j,k,l]*ε[k,l], (i,j))

    

def sachs_visc(a2,a4,Ecc,Eca):
    F = sachs_fluidity(a2,a4,Ecc,Eca)
    F_m = mandel.tensor2mandel(F)
    μ_m = ufl.inv(F_m)
    return mandel.mandel2tensor(μ_m)


def sachs_fluidity(a2,a4,Ecc,Eca):

   
    p1 = -(Ecc-1)/2
    p2 = (3*(Ecc-1) - 4*(Eca-1))/2
    p3 = Eca - 1

    D = a2.ufl_shape[0]

    I = ufl.Identity(D)
    I4 = ufl.as_tensor(0.5*(I[i,k]*I[j,l] + I[i,l]*I[j,k]), (i,j,k,l))
    
    # check anticommuter
    F = ufl.as_tensor((I4[i,j,k,l] + p2*a4[i,j,k,l] \
            + 0.5*p3*(a2[i,k]*I[j,l] + a2[i,l]*I[j,k] \
                      + I[i,k]*a2[l,j] + I[i,l]*a2[k,j])\
            + p1*I[i,j]*a2[k,l] ), (i,j,k,l) )
    
    return F/(0.4*Eca + 0.2*Ecc + 0.4)


