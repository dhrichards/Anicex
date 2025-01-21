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
import dolfinx_mpc
import nonlinear_mpc


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
        self.z.interpolate(lambda x: x[-1])

        self.z_original = fem.Function(self.Z)
        self.z_original.interpolate(lambda x: x[-1])

        

        

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


    def setup_stokes(self,FSSA = False, water_pressure = False):

        self.u = fem.Function(self.P2)
        self.p = fem.Function(self.P1)


        du, dp = ufl.TrialFunction(self.P2), ufl.TrialFunction(self.P1)
        v, q = ufl.TestFunction(self.P2), ufl.TestFunction(self.P1)

        με = lambda u: self.rheo.με(ε(u), self.f)
        η = lambda u: self.rheo.η(ε(u), self.f)

        


        n = ufl.FacetNormal(self.msh)
 

        if self.msh.geometry.dim ==2:
            f = fem.Constant(self.msh, default_scalar_type((
            self.params.ρ*self.params.g*ufl.sin(np.deg2rad(self.params.slope)),
            -self.params.ρ*self.params.g*ufl.cos(np.deg2rad(self.params.slope)))))
        else:
            f = fem.Constant(self.msh, default_scalar_type((
            self.params.ρ*self.params.g*ufl.sin(np.deg2rad(self.params.slope)),
            0.0,
            -self.params.ρ*self.params.g*ufl.cos(np.deg2rad(self.params.slope)))))
        
    
        F = [(η(self.u)*ufl.inner(με(self.u), ε(v)) \
        - ufl.inner(self.p, ufl.div(v)) \
        - ufl.inner(f, v)\
             )*ufl.dx,
        - ufl.inner(ufl.div(self.u), q) * ufl.dx ]

        if FSSA:
            F[0] -= self.params.dt*ufl.inner(self.u, n)*ufl.inner(f,v) * self.ds(1)

        if water_pressure:
            x = ufl.SpatialCoordinate(self.msh)
            z = x[self.msh.geometry.dim-1]
            pw = ufl.max_value(0.0,-self.params.ρ*self.params.g*z)
            F[0] += pw * ufl.inner(n, v) * self.ds

        J = get_jacobian(F,self.u,self.p,du,dp)
        P = get_preconditioner(J, self.u, dp, q, η)

        self.snes, self.x = nonlinear.nested_solve(F, J, self.u, self.p, self.bcs_u, P)
        # snes, x = nonlinear_mpc.nested_solve(F, J, self.u, self.p, self.bcs_u, mpc, mpc_q, P)
        # snes, x = nonlinear.block_solve(F, J, P, self.u, self.p, self.bcs_u, self.P2, self.P1)

    def solve_stokes(self):
        self.snes.solve(None, self.x)
        assert self.snes.getKSP().getConvergedReason() > 0

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()

    def setup_stokes_linear(self, periodic_boundary, periodic_relation):

        

        self.mpc_u = dolfinx_mpc.MultiPointConstraint(self.P2)
        self.mpc_u.create_periodic_constraint_geometrical(
            self.P2, periodic_boundary, periodic_relation, self.bcs_u)
        self.mpc_u.finalize()
        self.mpc_p = dolfinx_mpc.MultiPointConstraint(self.P1)
        self.mpc_p.finalize()

        self.u = fem.Function(self.mpc_u.function_space)
        self.p = fem.Function(self.mpc_p.function_space)

    def solve_stokes_linear(self):

        if self.msh.geometry.dim ==2:
            f = fem.Constant(self.msh, default_scalar_type((
            self.params.ρ*self.params.g*ufl.sin(np.deg2rad(self.params.slope)),
            -self.params.ρ*self.params.g*ufl.cos(np.deg2rad(self.params.slope)))))
        else:
            f = fem.Constant(self.msh, default_scalar_type((
            self.params.ρ*self.params.g*ufl.sin(np.deg2rad(self.params.slope)),
            0.0,
            -self.params.ρ*self.params.g*ufl.cos(np.deg2rad(self.params.slope)))))
        

        (u, p) = ufl.TrialFunction(self.P2), ufl.TrialFunction(self.P1)
        (v, q) = ufl.TestFunction(self.P2), ufl.TestFunction(self.P1)

        a00 = 2*self.rheo.B*ufl.inner(ε(u), ε(v))*ufl.dx
        a01 = -ufl.inner(p, ufl.div(v))*ufl.dx
        a10 = -ufl.inner(ufl.div(u), q) * ufl.dx
        a11 = None

        L0 = ufl.inner(f, v) * ufl.dx
        L1 = ufl.inner(dolfinx.fem.Constant(self.msh, default_scalar_type(0.0)), q) * ufl.dx

        a = [[dolfinx.fem.form(a00), dolfinx.fem.form(a01)], [dolfinx.fem.form(a10), dolfinx.fem.form(a11)]]
        L = [dolfinx.fem.form(L0), dolfinx.fem.form(L1)]

        P = dolfinx.fem.form(p * q * ufl.dx)

        self.ksp, self.x, self.b = nonlinear_mpc.linear_nested_solver(
            a, L, self.bcs_u, P, self.mpc_u, self.mpc_p)
        
    
    # def solve_stokes_linear(self):
        self.ksp.solve(self.b, self.x)
        assert self.ksp.getConvergedReason() > 0

        self.u.x.petsc_vec.setArray(self.x.getNestSubVecs()[0].array)
        self.p.x.petsc_vec.setArray(self.x.getNestSubVecs()[1].array)

        self.u.x.scatter_forward()
        self.p.x.scatter_forward()

        self.mpc_u.backsubstitution(self.u)
        self.mpc_p.backsubstitution(self.p)
    


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


    def move_mesh(self, periodic_boundary=None, periodic_relation=None, k = 1e-4):

        D = self.msh.geometry.dim
        
        if periodic_boundary is not None:
            mpc = dolfinx_mpc.MultiPointConstraint(self.Z)
            mpc.create_periodic_constraint_geometrical(
                self.Z, periodic_boundary, periodic_relation, self.bc_z)
            mpc.finalize()


        z = ufl.TrialFunction(self.Z)
        v = ufl.TestFunction(self.Z)
        
        ds = self.ds(1) # top boundary
        
        acc = self.params.acc(ufl.SpatialCoordinate(self.msh)) # accumulation rate
        
        if D == 2:
            rh = integrators.RK4(
                lambda z: self.u[D-1] + acc - self.dt*ufl.dot(self.u[0], ufl.grad(z)[0]),
                self.z, self.dt)
        else:
            rh = integrators.RK4(
                lambda z: self.u[D-1] + acc - self.dt*ufl.dot(self.u[0], ufl.grad(z)[0]) \
                    - self.dt*ufl.dot(self.u[1], ufl.grad(z)[1]),
                self.z, self.dt)

        # Basically trying to solve the surface equation on ds, 
        # and an equation the dz/dz is constant, i.e. d2z/dz2 = 0 in the domain
        a = ufl.inner(z,v)*ds + k*self.dt*ufl.inner(ufl.Dx(z,D-1),ufl.Dx(v,D-1))*ufl.dx
        # +   k*self.dt*ufl.inner(ufl.grad(z), ufl.grad(v))*ufl.dx
            
    
        L = ufl.inner(rh,v)*ds
# 
        if periodic_boundary is not None:
            problem = dolfinx_mpc.LinearProblem(a, L, bcs=[], mpc=mpc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        else:
            problem = fem.petsc.LinearProblem(a, L, bcs=self.bc_z, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.z = problem.solve()


        self.msh.geometry.x[:,D-1] = self.z.x.array.real








def get_jacobian(F,u,p,du,dp):
    return [[ufl.derivative(F[0], u, du), ufl.derivative(F[0], p, dp)],
            [ufl.derivative(F[1], u, du), ufl.derivative(F[1], p, dp)]]

def get_preconditioner(J, u, dp, q, η):
    return [[J[0][0], None],
            [None, (2 * η(u))**-1 * ufl.inner(dp, q) * ufl.dx]]
