import ufl
import mandel
from ufl import i,j,k,l


def sachs_τhat(ε,a2,a4,Ecc,Eca):
    μ = sachs_visc(a2,a4,Ecc,Eca)
    ι = Eca/(0.4*Eca + 0.2*Ecc + 0.4)
    return ι*(ufl.as_tensor(μ[i,j,k,l]*ε[k,l], (i,j)))

    

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
    
    # check anticommuter
    F = ufl.as_tensor((I[i,k]*I[j,l] + p2*a4[i,j,k,l] \
            + 0.5*p3*(I[i,k]*a2[j,l] + a2[i,k]*I[j,l] + ...)
            + p1*I[i,j]*a2[k,l]), (i,j,k,l) )
    
    return F/(0.4*Eca + 0.2*Ecc + 0.4)


