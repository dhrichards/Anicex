import dolfinx.nls.petsc
import numpy as np
from dolfinx import fem, default_real_type, default_scalar_type
import basix.ufl as bufl
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import nonlinear
import specfabufl
import integrators

def ε(u):
    return ufl.sym(ufl.grad(u))


class solver:
    def __init__(self, msh, bc_funcs, params, rheo, fabric):
        self.msh = msh
        self.bc_funcs = bc_funcs
        self.params = params
        self.rheo = rheo
        self.fabric = fabric
        self.dt = params.dt

        

        P2_el = bufl.element("Lagrange", self.msh.basix_cell(), 2, shape=(self.msh.geometry.dim,))
        P1_el = bufl.element("Lagrange", self.msh.basix_cell(), 1)

        self.P2 = fem.functionspace(self.msh, P2_el)
        self.P1 = fem.functionspace(self.msh, P1_el)

        self.Z = fem.functionspace(msh, ("Lagrange", 1))

        self.bcs_u = bc_funcs[0](self.P2)

        self.u = fem.Function(self.P2)
        self.p = fem.Function(self.P1)

        f_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, shape=(specfabufl.nlm_len_from_L(params.L),))
      
        # self.F = fem.functionspace(self.msh, bufl.mixed_element([f_el, f_el]))
        # self.f = fem.Function(self.F)
        # self.fr = self.f.sub(0); self.fi = self.f.sub(1)
        # self.fr.interpolate(self.fabric.f0)

        self.F = fem.functionspace(self.msh, f_el)
        self.f = fem.Function(self.F)
        self.f.interpolate(self.fabric.f0)

        self.bcs_f = bc_funcs[1](self.F)

    def solve_stokes(self):
        du, dp = ufl.TrialFunction(self.P2), ufl.TrialFunction(self.P1)
        v, q = ufl.TestFunction(self.P2), ufl.TestFunction(self.P1)

        με = lambda u: self.rheo.με(ε(u), self.f)
        η = lambda u: self.rheo.η(ε(u), self.f)
        
        n = ufl.FacetNormal(self.msh)
        ds = ufl.Measure("ds", domain=self.msh)

        f = fem.Constant(self.msh, default_scalar_type((
            self.params.ρ*self.params.g*ufl.sin(np.deg2rad(self.params.slope)),
            -self.params.ρ*self.params.g*ufl.cos(np.deg2rad(self.params.slope)))))
        
    
        F = [(η(self.u)*ufl.inner(με(self.u), ε(v)) \
        - ufl.inner(self.p, ufl.div(v)) \
        - ufl.inner(f, v)\
             )*ufl.dx,
        - ufl.inner(ufl.div(self.u), q) * ufl.dx ]


        test1 = ufl.derivative(F[0], self.u, du)

        test = fem.form(test1)

        J = get_jacobian(F,self.u,self.p,du,dp)
        P = get_preconditioner(J, self.u, dp, q, η)

        snes, x = nonlinear.nested_solve(F, J, self.u, self.p, self.bcs_u, P)
        # snes, x = nonlinear.block_solve(F, J, P, self.u, self.p, self.bcs_u, self.P2, self.P1)

        snes.solve(None, x)
        assert snes.getKSP().getConvergedReason() > 0

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()


    def solve_fabric_real(self, k = 1e-4):


        (fr, fi) = ufl.TrialFunctions(self.F)
        (gr, gi) = ufl.TestFunctions(self.F)

        # M = self.fabric.M(self.u, self.f, self.rheo)
        M = specfabufl.M_CDRX(4)
        Mr = ufl.real(M); Mi = M

        # rh = ufl.dot(M, fr)

        a = ( ufl.inner(fr,gr) \
            + self.dt*ufl.inner(ufl.dot(ufl.grad(fr), self.u), gr) \
            + self.dt*k*ufl.inner(ufl.grad(fr), ufl.grad(gr)) \
            - self.dt*ufl.inner(ufl.dot(Mr,fr) - ufl.dot(Mi,fi),gr)\
            # - self.dt*ufl.inner(rh,gr)\
                        )*ufl.dx
        
        a += ( ufl.inner(fi,gi) \
              + self.dt*ufl.inner(ufl.dot(ufl.grad(fi), self.u), gi) \
              + self.dt*k*ufl.inner(ufl.grad(fi), ufl.grad(gi)) \
                # + self.dt*ufl.inner(ufl.dot(Mi,fr) + ufl.dot(Mr,fi),gi)  \
                    )*ufl.dx
        

        L = (ufl.inner(self.fr,gr) + ufl.inner(self.fi,gi))*ufl.dx

        problem = fem.petsc.LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

        self.f = problem.solve()

    def solve_fabric_complex(self, k = 1e-4):

        f = ufl.TrialFunction(self.F)
        g = ufl.TestFunction(self.F)

        # M = self.fabric.M(self.u, self.f, self.rheo)
        M = specfabufl.M_LROT(4, ε(self.u), ufl.skew(ufl.grad(self.u)), 1.0)\
            + 0.1*specfabufl.M_CDRX(4)
        # M = specfabufl.M_CDRX(4)


        # rh = ufl.dot(M, f)

        a = (ufl.inner(f, g) \
            # + self.dt*ufl.inner(ufl.dot(ufl.grad(f), self.u), g) \
            # + self.dt*k*ufl.inner(ufl.grad(f), ufl.grad(g)) \
            # - self.dt*ufl.inner(ufl.dot(M, f), g)\
                        )*ufl.dx
        
        L = (ufl.inner(integrators.ForwardEuler(M,self.f,self.dt), g) \
            #  + self.dt*ufl.inner(ufl.dot(M,self.f),g)\
                )*ufl.dx

        problem = fem.petsc.LinearProblem(a, L, bcs=[],petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.f = problem.solve()


    def move_mesh(self, k = 1e-4):

        D = self.msh.geometry.dim

        z = ufl.TrialFunction(self.Z)
        v = ufl.TestFunction(self.Z)

        us = self.u[:D-1]; w = self.u[-1]

        grad_s = lambda u: ufl.as_vector([ufl.grad(u)[i] for i in range(D-1)])

        ds = ufl.Measure("ds", domain=self.msh)

        a = (ufl.inner(z, v) + self.dt*ufl.inner(us, grad_s(z)))*ds \
            + k*ufl.inner(ufl.grad(z), ufl.grad(v))*ufl.dx
    
        L = ufl.inner(self.dt*(w+self.params.acc), v)*ds

        problem = fem.LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        dz = problem.solve()

        self.msh.geometry.x[:,D-1] += self.params.dt*dz.x.array








def get_jacobian(F,u,p,du,dp):
    return [[ufl.derivative(F[0], u, du), ufl.derivative(F[0], p, dp)],
            [ufl.derivative(F[1], u, du), ufl.derivative(F[1], p, dp)]]

def get_preconditioner(J, u, dp, q, η):
    return [[J[0][0], None],
            [None, (2 * η(u))**-1 * ufl.inner(dp, q) * ufl.dx]]
