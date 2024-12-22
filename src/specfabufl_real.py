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

        M_lrot = M_LROT(self.L,self.C(u,f,rheo),self.W(u),1.0)
        M_reg = M_REG(self.L,self.C(u,f,rheo))
        M_rrx = M_CDRX(self.L)
        
        return M_lrot + self.λ*self.SR(u)*M_rrx #+ M_reg
    

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


def M_LROT(Lmax,C,W,ι):
    
    qe_r,qe_i = quad_rr(C*ι)
    qo_r,qo_i = quad_tp(W)

    g0_rot_r = as_vector([0,0,0.0,0,0,0])
    gz_rot_r = as_vector([1*sqrt(3)*qo_i[1],0,0,0,0,0])
    gn_rot_r = as_vector([6/sqrt(6)*qo_i[0],0,0,0,0,0])
    gp_rot_r = as_vector([-6/sqrt(6)*qo_i[2],0,0,0,0,0])

    g0_rot_i = as_vector([0,0,0.0,0,0,0])
    gz_rot_i = as_vector([-1*sqrt(3)*qo_r[1],0,0,0,0,0])
    gn_rot_i = as_vector([-6/sqrt(6)*qo_r[0],0,0,0,0,0])
    gp_rot_i = as_vector([6/sqrt(6)*qo_r[2],0,0,0,0,0])

    g0_Tay_r = as_vector([0,3*qe_r[0],3*qe_r[1],3*qe_r[2],3*qe_r[3],3*qe_r[4]])
    gz_Tay_r = as_vector([0,-qe_r[0],0,0,0,qe_r[4]])
    gn_Tay_r = as_vector([sqrt(5/6)*qe_r[1],0,qe_r[0],sqrt(2/3)*qe_r[1],sqrt(3/2)*qe_r[2],2*qe_r[3]])
    gp_Tay_r = as_vector([sqrt(5/6)*qe_r[3],2*qe_r[1],sqrt(3/2)*qe_r[2],sqrt(2/3)*qe_r[3],qe_r[4],0])

    g0_Tay_i = as_vector([0,3*qe_i[0],3*qe_i[1],3*qe_i[2],3*qe_i[3],3*qe_i[4]])
    gz_Tay_i = as_vector([0,-qe_i[0],0,0,0,qe_i[4]])
    gn_Tay_i = as_vector([sqrt(5/6)*qe_i[1],0,qe_i[0],sqrt(2/3)*qe_i[1],sqrt(3/2)*qe_i[2],2*qe_i[3]])
    gp_Tay_i = as_vector([sqrt(5/6)*qe_i[3],2*qe_i[1],sqrt(3/2)*qe_i[2],sqrt(2/3)*qe_i[3],qe_i[4],0])

    g0_r = g0_rot_r + g0_Tay_r
    gz_r = gz_rot_r + gz_Tay_r
    gn_r = gn_rot_r + gn_Tay_r
    gp_r = gp_rot_r + gp_Tay_r

    g0_i = g0_rot_i + g0_Tay_i
    gz_i = gz_rot_i + gz_Tay_i
    gn_i = gn_rot_i + gn_Tay_i
    gp_i = gp_rot_i + gp_Tay_i


    nlm_len = nlm_len_from_L(Lmax)
    
    i,j,k = ufl.indices(3)

    M_LROT_r = -1.0*as_tensor(GC[i,j,k]*g0_r[k] + GCm[i,j,k]*gz_r[k] + GC_m1[i,j,k]*gn_r[k] + GC_p1[i,j,k]*gp_r[k],(i,j))
    M_LROT_i = -1.0*as_tensor(GC[i,j,k]*g0_i[k] + GCm[i,j,k]*gz_i[k] + GC_m1[i,j,k]*gn_i[k] + GC_p1[i,j,k]*gp_i[k],(i,j))

    return M_LROT_r,M_LROT_i

def M_CDRX(Lmax):

    l,m = get_lm(Lmax)
    Ldiag = -l*(l+1)


    return ufl.diag(as_vector(Ldiag))


def M_REG(Lmax,C):

    expo = 1.700
    nu = 1.9879322126397958

    l,m = get_lm(Lmax)
    Ldiag = -l*(l+1)

    Reg_diag = abs(Ldiag/(Lmax*(Lmax+1)))**expo

    M = ufl.diag(as_vector(Reg_diag))

    ratemag = nu*ufl.sqrt(ufl.tr(ufl.dot(C,C)))

    return -ratemag*M


def quad_rr(M):
    """
    Compute the quadrupole representation `quad_rr` of the UFL tensor M.

    Parameters:
        M (ufl.Tensor): A 3x3 UFL tensor.

    Returns:
        ufl.Vector: The result as a 5-element UFL vector.
    """
    fsq = sqrt(2 * pi / 15)

    r = as_vector([
        fsq * (M[0, 0] - M[1, 1]),
        2 * fsq * M[0, 2],
        -2 / 3 * sqrt(pi / 5) * (M[0, 0] + M[1, 1] - 2 * M[2, 2]),
        -2 * fsq * M[0, 2],
        fsq * (M[0, 0] - M[1, 1])
    ])

    i = as_vector([
        fsq * 2 * M[0, 1],
        2 * fsq * M[1, 2],
        0,
        2 * fsq * M[1, 2],
        fsq * -2 * M[0, 1]
    ])

    return r, i

def quad_tp(M):
    """
    Compute the quadrupole representation `quad_tp` of the UFL tensor M.

    Parameters:
        M (ufl.Tensor): A 3x3 UFL tensor.

    Returns:
        ufl.Vector: The result as a 3-element UFL vector.
    """
    fsq = sqrt(2 * pi / 3)
    r = fsq * as_vector([
            M[1, 2],
            sqrt(2) * M[0, 1],
            -M[1, 2]
    ])

    i = fsq * as_vector([
            -M[0, 2],
            0,
            -M[0, 2]
    ])

    
    return r, i


def lm2idx(l,m):
    return int((l*(l+1))/2 + m)


