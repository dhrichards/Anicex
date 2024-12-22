import dolfinx.nls.petsc
import numpy as np
from dolfinx import fem, default_real_type, default_scalar_type, mesh
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
    def __init__(self, msh, bc_funcs, params, rheo, fabric, top):
        self.msh = msh
        self.bc_funcs = bc_funcs
        self.params = params
        self.rheo = rheo
        self.fabric = fabric
        self.dt = params.dt

        top_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, top)
        mesh_tags = mesh.meshtags(msh, msh.topology.dim - 1, top_facets, 1)
        self.ds = ufl.Measure("ds", domain=msh, subdomain_data=mesh_tags)


        

        P2_el = bufl.element("Lagrange", self.msh.basix_cell(), 2, shape=(self.msh.geometry.dim,))
        P1_el = bufl.element("Lagrange", self.msh.basix_cell(), 1)

        self.P2 = fem.functionspace(self.msh, P2_el)
        self.P1 = fem.functionspace(self.msh, P1_el)

        self.Z = fem.functionspace(msh, ("Lagrange", 1))
        self.z = fem.Function(self.Z)
        self.z.interpolate(lambda x: x[1])

        

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


        self.bcs_u = bc_funcs[0](self.P2)
        self.bcs_f = bc_funcs[1](self.F)
        self.bc_z = bc_funcs[2](self.Z)

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

    def solve_fabric_complex(self, k = 1e-4):

        f = ufl.TrialFunction(self.F)
        g = ufl.TestFunction(self.F)

        M = self.fabric.M(self.u, self.f, self.rheo)
        # M = specfabufl.M_LROT(4, ε(self.u), ufl.skew(ufl.grad(self.u)), 1.0)\
        #     + 0.1*specfabufl.M_CDRX(4)
        

        rh = integrators.SSPRK43(
            lambda f: ufl.dot(M, f), self.f, self.dt)
        

        a = (ufl.inner(f, g) \
            + self.dt*ufl.inner(ufl.dot(ufl.grad(f), self.u), g) \
            + self.dt*k*ufl.inner(ufl.grad(f), ufl.grad(g)) \
            # - self.dt*ufl.inner(ufl.dot(M, f), g)\
                        )*ufl.dx
        
        L = ufl.inner(rh, g)*ufl.dx

        problem = fem.petsc.LinearProblem(a, L, bcs=[],petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.f = problem.solve()


    def move_mesh(self, k = 1e-4):

        D = self.msh.geometry.dim

        z = ufl.TrialFunction(self.Z)
        v = ufl.TestFunction(self.Z)

        n = ufl.FacetNormal(self.msh)

        w = ufl.dot(self.u, n)
        us = self.u - w*n

        # grad_s = lambda u: ufl.grad(u)[0]

        ds = self.ds(1)
        
        a = self.params.acc

        rh = integrators.RK4(
            lambda z: w + a - self.dt*ufl.dot(us, ufl.grad(z)),
            self.z, self.dt)

        a = ufl.inner(z,v)*ds +\
              k*self.dt*ufl.inner(ufl.grad(z), ufl.grad(v))*ufl.dx
    
        L = ufl.inner(rh,v)*ds
# 
        problem = fem.petsc.LinearProblem(a, L, bcs=self.bc_z, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.z = problem.solve()


        # self.msh.geometry.x[:,D-1] += self.params.dt*dz.x.array








def get_jacobian(F,u,p,du,dp):
    return [[ufl.derivative(F[0], u, du), ufl.derivative(F[0], p, dp)],
            [ufl.derivative(F[1], u, du), ufl.derivative(F[1], p, dp)]]

def get_preconditioner(J, u, dp, q, η):
    return [[J[0][0], None],
            [None, (2 * η(u))**-1 * ufl.inner(dp, q) * ufl.dx]]
