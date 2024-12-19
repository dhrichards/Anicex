import ufl
from invariants import eigenstate
from Rheology import mandel, sachs



j = [1,2,0]
k = [2,0,1]

def orthotropic_full(ε,a2,a4,Ecc,Eca,α = 0.0125, n=3.0):

    eigvals, eigproj = eigenstate(a2)
    #eigproj ij = eigvec i * eigvec j
    m = [ufl.as_vector([eigproj[i][j] for j in range(3)]) for i in range(3)]

    E = Eij_tranisotropic(m,a2,a4,Ecc,Eca,α)

    λi, λi3, γ = λγ_Full(E,n)

    # Mi, Mi3 = M_matrices(m)
    Ii, Ii3 = Invariants(ε,m)
    I = ufl.Identity(3)

    με = sum([( (λi[i]/γ)*(Ii[j[i]]-Ii[k[i]])*(I-3*ufl.outer(m[i],m[i])/2)\
               + (4/λi3[i])*Ii3[i]*0.5*(ufl.outer(m[j[i]],m[k[i]])+ufl.outer(m[k[i]],m[j[i]])) ) for i in range(3)])
    

    η = (sum([( (λi[i]/γ)*(Ii[j[i]]-Ii[k[i]])**2 + (4/λi3[i])*Ii3[i]**2 ) for i in range(3)]))**((1-n)/(2*n))

    return η, με


def Invariants(ε,m):
    Mi, Mi3 = M_matrices(m)

    Ii = [ufl.inner(ε,Mi[i]) for i in range(3)]
    Ii3 = [ufl.inner(ε,Mi3[i]) for i in range(3)]

    return Ii, Ii3



def M_matrices(m):
    Mi = [(ufl.outer(m[j[i]],m[j[i]]) - ufl.outer(m[k[i]],m[k[i]]))/2  for i in range(3)]
    Mi3 = [(ufl.outer(m[j[i]],m[k[i]]) + ufl.outer(m[k[i]],m[j[i]]))/2  for i in range(3)]

    return Mi, Mi3


def λγ_Full(E,n):

    p = 2/(n+1)

    # γ = 2*(E22*E33)**nn + 2*(E33*E11)**nn + 2*(E11*E22)**nn - E11**(2*nn) - E22**(2*nn) - E33**(2*nn)         
    γ = 2*(E[1,1]*E[2,2])**p + 2*(E[2,2]*E[0,0])**p + 2*(E[0,0]*E[1,1])**p - E[0,0]**(2*p) - E[1,1]**(2*p) - E[2,2]**(2*p)


    λi = [(4/3)*(E[j[i],j[i]]**p + E[k[i],k[i]]**p - E[i,i]**p) for i in range(3)]
    λi3 = [2*E[j[i],k[i]]**p for i in range(3)]
    return λi, λi3, γ


def Eij_tranisotropic(m,a2,a4,Ecc,Eca,α):

    E1  = Evw_tranisotropic(m[0],m[0],τ_vv(m[0]),a2,a4,Ecc,Eca,α)
    E2  = Evw_tranisotropic(m[1],m[1],τ_vv(m[1]),a2,a4,Ecc,Eca,α)
    E3  = Evw_tranisotropic(m[2],m[2],τ_vv(m[2]),a2,a4,Ecc,Eca,α)
    E4  = Evw_tranisotropic(m[1],m[2],τ_vw(m[1],m[2]),a2,a4,Ecc,Eca,α)
    E5  = Evw_tranisotropic(m[0],m[2],τ_vw(m[0],m[2]),a2,a4,Ecc,Eca,α)
    E6  = Evw_tranisotropic(m[0],m[1],τ_vw(m[0],m[1]),a2,a4,Ecc,Eca,α)

    return ufl.as_matrix([[E1, E6, E5],
                            [E6, E2, E4],
                            [E5, E4, E3]])


def Evw_tranisotropic(v,w,τ,a2,a4,Ecc,Eca,α):

    I = ufl.Identity(3)
    a2i = I/3

    i,j,k,l = ufl.indices(4)

    a4i = ufl.as_tensor((I[i,j]*I[k,l] + I[i,k]*I[j,l] + I[i,l]*I[j,k])/15, (i,j,k,l))

    vw = ufl.outer(v,w)

    Evw_sachs = ufl.inner(sachshomo(τ,a2,a4,Ecc,Eca),vw)/ufl.inner(sachshomo(τ,a2i,a4i,Ecc,Eca),vw)
    Evw_taylor = ufl.inner(taylorhomo(τ,a2,a4,Ecc,Eca,α),vw)/ufl.inner(taylorhomo(τ,a2i,a4i,Ecc,Eca,α),vw)

    return (1-α)*Evw_sachs + α*Evw_taylor


def τ_vv(v):
    return ufl.Identity(3)/3 - ufl.outer(v,v)

def τ_vw(v,w):
    return ufl.outer(v,w) + ufl.outer(w,v)


def sachshomo(τ,a2,a4,Ecc,Eca):
    p1 = -(Ecc-1)/2
    p2 = (3*(Ecc-1) - 4*(Eca-1))/2
    p3 = Eca - 1

    I = ufl.Identity(3)

    i,j,k,l = ufl.indices(4)

    return ufl.as_tensor((τ[i,j] + p1*(a2[k,l]*τ[k,l])*I[i,j] + p2*a4[i,j,k,l]*τ[k,l] + p3*(a2[i,k]*τ[k,j] + τ[i,k]*a2[k,j])), (i,j))

def taylorhomo(τ,a2,a4,Ecc,Eca):
    μ = sachs.sachs_fluidity(a2,a4,1/Ecc,1/Eca)
    F = mandel.mandel2tensor(ufl.inv(mandel.tensor2mandel(μ)))

    i, j, k, l = ufl.indices(4)
    return ufl.as_tensor(F[i,j,k,l]*τ[k,l], (i,j))

