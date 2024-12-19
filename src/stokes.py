import dolfinx.nls.petsc
import numpy as np
from dolfinx import fem, default_real_type, default_scalar_type
import basix.ufl as bufl
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import nonlinear
import specfabufl

def ε(u):
    return ufl.sym(ufl.grad(u))


class solver:
    def __init__(self, msh, bc_funcs, params, rheo, fabric):
        self.msh = msh
        self.bc_funcs = bc_funcs
        self.params = params
        self.rheo = rheo
        self.fabric = fabric

        f_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, shape=(specfabufl.nlm_len_from_L(params.L),), dtype=default_real_type)
        
        self.F = fem.functionspace(self.msh, f_el)

        P2_el = bufl.element("Lagrange", self.msh.basix_cell(), 2, shape=(self.msh.geometry.dim,), dtype=default_real_type)
        P1_el = bufl.element("Lagrange", self.msh.basix_cell(), 1, dtype=default_real_type)

        self.P2 = fem.functionspace(self.msh, P2_el)
        self.P1 = fem.functionspace(self.msh, P1_el)

        self.Z = fem.functionspace(msh, ("Lagrange", 1))

        self.bcs_u = bc_funcs[0](self.P2)
        self.bcs_f = bc_funcs[1](self.F)

        self.u = fem.Function(self.P2)
        self.p = fem.Function(self.P1)
        self.f = fem.Function(self.F)

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
        - ufl.inner(f, v) )*ufl.dx,
        - ufl.inner(ufl.div(self.u), q) * ufl.dx ]

        J = get_jacobian(F,self.u,self.p,du,dp)
        P = get_preconditioner(J, self.u, dp, q, η)

        snes, x = nonlinear.nested_solve(F, J, self.u, self.p, self.bcs_u, P)
        # snes, x = nonlinear.block_solve(F, J, P, self.u, self.p, self.bcs_u, self.P2, self.P1)

        snes.solve(None, x)
        assert snes.getKSP().getConvergedReason() > 0

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()


    def solve_fabric(self, k = 1e-4):


        f = ufl.TrialFunction(self.F)
        g = ufl.TestFunction(self.F)

        M = self.fabric.M(self.u, self.f, self.rheo)

        a = ( ufl.inner(f,g) \
            + self.dt*ufl.inner(ufl.inner(self.uh, ufl.grad(f)), g) \
            + self.dt*k*ufl.inner(ufl.grad(f), ufl.grad(g)) \
            - self.dt*ufl.dot(ufl.dot(M,f),g)    )*ufl.dx
        
        L = ufl.inner(self.f,g)*ufl.dx

        problem = fem.LinearProblem(a, L, bcs=self.bcs_f, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

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
            [None, (2 * η(u))**-1 * dp * q * ufl.dx]]
