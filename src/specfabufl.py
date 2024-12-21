from ufl import sqrt, pi, as_tensor, as_vector
import gauntbody as gb
import ufl
import numpy as np

GC = ufl.as_tensor(gb.GC[:,:,:6])
GCm = ufl.as_tensor(gb.GCm[:,:,:6])
GC_m1 = ufl.as_tensor(gb.GC_m1[:,:,:6])
GC_p1 = ufl.as_tensor(gb.GC_p1[:,:,:6])

class fabric:
    def __init__(self,L,λ,αD,αS,β):
        self.L = L
        self.λ = λ
        self.αD = αD
        self.αS = αS
        self.β = β

        self.nlm_len = nlm_len_from_L(L)



    
    def W(self,u):
        return ufl.skew(ufl.grad(u))
    
    def D(self,u):
        return ufl.sym(ufl.grad(u))
    
    def SR(self,u):
        return ufl.sqrt(0.5*ufl.inner(self.D(u),self.D(u)))
    
    def C(self,u,f,rheo):
        return self.αD*self.D(u) + self.αS*rheo.τhat(self.D(u),f)
    
    def M(self,u,f,rheo):
        
        return makeM(self.W(u),self.C(u,f,rheo),self.SR(u),self.λ,self.L)
    

    def f0(self,x):
        vals = np.zeros((self.nlm_len, x.shape[1]), dtype=np.complex128)
        vals[0] = 1.0/np.sqrt(4*np.pi)

        return vals
    

    

    


def nlm_len_from_L(Lmax):
    return int(((Lmax+1)*(Lmax+2))/2)


def get_lm(Lmax):
    l = []; m = []
    for ll in range(0,Lmax+1,2):
        for mm in range(-ll,ll+1):
            l.append(ll)
            m.append(mm)
    return np.array(l),np.array(m)


def makeM(W,C,SR,λ,L):

    M_lrot = M_LROT(L,C,W,1.0)
    M_reg = M_REG(L,C)
    M_rrx = M_CDRX(L)

    return M_lrot + M_reg + λ*SR*M_rrx

def M_LROT(Lmax,C,W,ι):
    
    qe = quad_rr(C*ι)
    qo = quad_tp(W)

    

    g0_rot = as_vector([0,0,0.0,0,0,0])
    gz_rot = as_vector([-1j*sqrt(3)*qo[1],0,0,0,0,0])
    gn_rot = as_vector([-1j*6/sqrt(6)*qo[0],0,0,0,0,0])
    gp_rot = as_vector([1j*6/sqrt(6)*qo[2],0,0,0,0,0])

    g0_Tay = as_vector([0,3*qe[0],3*qe[1],3*qe[2],3*qe[3],3*qe[4]])
    gz_Tay = as_vector([0,-qe[0],0,0,0,qe[4]])
    gn_Tay = as_vector([sqrt(5/6)*qe[1],0,qe[0],sqrt(2/3)*qe[1],sqrt(3/2)*qe[2],2*qe[3]])
    gp_Tay = as_vector([sqrt(5/6)*qe[3],2*qe[1],sqrt(3/2)*qe[2],sqrt(2/3)*qe[3],qe[4],0])

    g0 = g0_rot + g0_Tay
    gz = gz_rot + gz_Tay
    gn = gn_rot + gn_Tay
    gp = gp_rot + gp_Tay


    nlm_len = nlm_len_from_L(Lmax)
    
    i,j,k = ufl.indices(3)
    # M_LROT = as_tensor(
    #     [
    #         [
    #             -(
    #                 sum(GC[i, j, k] * g0[k] for k in range(6)) +
    #                 sum(GCm[i, j, k] * gz[k] for k in range(6)) +
    #                 sum(GC_m1[i, j, k] * gn[k] for k in range(6)) +
    #                 sum(GC_p1[i, j, k] * gp[k] for k in range(6))
    #             )
    #             for j in range(nlm_len)
    #         ]
    #         for i in range(nlm_len)
    #     ]
    # )
    M_LROT = -1.0*as_tensor(GC[i,j,k]*g0[k] + GCm[i,j,k]*gz[k] + GC_m1[i,j,k]*gn[k] + GC_p1[i,j,k]*gp[k],(i,j))

    return M_LROT

def M_CDRX(Lmax):

    l,m = get_lm(Lmax)
    Ldiag = -l*(l+1)


    return ufl.diag(as_vector(Ldiag))


def M_REG(Lmax,C):

    expo = 1.700
    nu = 1.9879322126397958

    l,m = get_lm(Lmax)
    Ldiag = l*(l+1)

    Reg_diag = np.abs(Ldiag/(Lmax*(Lmax+1)))**expo

    ratemag = nu*ufl.sqrt(ufl.inner(C,C))

    return ufl.diag(as_vector(Reg_diag)*ratemag)


def quad_rr(M):
    """
    Compute the quadrupole representation `quad_rr` of the UFL tensor M.

    Parameters:
        M (ufl.Tensor): A 3x3 UFL tensor.

    Returns:
        ufl.Vector: The result as a 5-element UFL vector.
    """
    fsq = sqrt(2 * pi / 15)
    return as_vector([
        fsq * (M[0, 0] - M[1, 1] + 2 * 1j * M[0, 1]),
        2 * fsq * (M[0, 2] + 1j * M[1, 2]),
        -2 / 3 * sqrt(pi / 5) * (M[0, 0] + M[1, 1] - 2 * M[2, 2]),
        -2 * fsq * (M[0, 2] - 1j * M[1, 2]),
        fsq * (M[0, 0] - M[1, 1] - 2 * 1j * M[0, 1])
    ])

def quad_tp(M):
    """
    Compute the quadrupole representation `quad_tp` of the UFL tensor M.

    Parameters:
        M (ufl.Tensor): A 3x3 UFL tensor.

    Returns:
        ufl.Vector: The result as a 3-element UFL vector.
    """
    fsq = sqrt(2 * pi / 3)
    return fsq * as_vector([
        M[1, 2] - 1j * M[0, 2],
        sqrt(2) * M[0, 1],
        -M[1, 2] - 1j * M[0, 2]
    ])


def lm2idx(l,m):
    return int((l*(l+1))/2 + m)


