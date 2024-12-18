import dolfinx.nls.petsc
import numpy as np
from dolfinx import fem, default_real_type, default_scalar_type
import basix.ufl as bufl
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from nonlinear import NonlinearPDE_SNESProblem
from Rheology import sachs



def solve_no_damage(mesh, bc_func, material, dt, u=None, p=None):

    C1 = material.C1; C2 = material.C2
    ρw = material.ρw; ρi = material.ρi

    P2_el = bufl.element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,), dtype=default_real_type)
    P1_el = bufl.element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)

    P2 = fem.functionspace(mesh, P2_el)
    P1 = fem.functionspace(mesh, P1_el)

    bcs = bc_func(P2)

    if u is None:
        u = fem.Function(P2)
    if p is None:
        p = fem.Function(P1)

    du, dp = ufl.TrialFunction(P2), ufl.TrialFunction(P1)
    v, q = ufl.TestFunction(P2), ufl.TestFunction(P1)

    def η(u):
        return viscosity(u, material.n, 1.e-8)
    
    def τ(u,μ):
        return ufl.as_matrix(μ[ufl.i,ufl.j,ufl.k,ufl.l]*ε(u)[ufl.k,ufl.l],(ufl.i,ufl.j))
    

    if mesh.geometry.dim == 2:
        f = fem.Constant(mesh, default_scalar_type((0, -ρi/ρw)))
    else:
        f = fem.Constant(mesh, default_scalar_type((0, 0, -ρi/ρw)))


    # Outward-pointing unit normal to the boundary  
    n = ufl.FacetNormal(mesh)           

    # Surface measure
    ds = ufl.Measure("ds", domain=mesh)

    # Water pressure
    # pw = water_pressure(mesh,vh,material)
    def pw(u):
        return water_pressure(mesh,u*dt)
    # pw = water_pressure_static(mesh)


    
    F = [(1/C2)*η(u)*ufl.inner(τ(u,μ), ε(v)) * ufl.dx \
        - ufl.inner(p, ufl.div(v)) * ufl.dx \
        - C1 * ufl.inner(f, v) * ufl.dx \
        + C1*pw(u)*ufl.inner(n, v) * ds,
        - ufl.inner(ufl.div(u), q) * ufl.dx ]
    

    J = get_jacobian(F,u,p,du,dp)
    P = get_preconditioner(J, u, dp, q, η) 
    
    
    return _nested_solve(F, J, P, u, p, bcs)
    

def get_jacobian(F,u,p,du,dp):
    return [[ufl.derivative(F[0], u, du), ufl.derivative(F[0], p, dp)],
            [ufl.derivative(F[1], u, du), ufl.derivative(F[1], p, dp)]]

def get_preconditioner(J, u, dp, q, η):
    return [[J[0][0], None],
            [None, (2 * η(u))**-1 * dp * q * ufl.dx]]


def _nested_solve(F, J, P, u, p, bcs):
    F, J, P = fem.form(F), fem.form(J), fem.form(P)


    Jmat = fem.petsc.create_matrix_nest(J)
    Pmat = fem.petsc.create_matrix_nest(P)
    Fvec = fem.petsc.create_vector_nest(F)

    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    snes.setTolerances(rtol=1.0e-15, max_it=10)
    nested_IS = Jmat.getNestISs()
    snes.getKSP().setType("minres")
    snes.getKSP().setTolerances(rtol=1e-12)
    snes.getKSP().getPC().setType("fieldsplit")
    snes.getKSP().getPC().setFieldSplitIS(["u", nested_IS[0][0]], ["p", nested_IS[1][1]])

    snes.getKSP().getPC().setFieldSplitType(
                PETSc.PC.CompositeType.ADDITIVE)

    ksp_u, ksp_p = snes.getKSP().getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("hypre")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("hypre")

    problem = NonlinearPDE_SNESProblem(F, J, [u, p], bcs=bcs, P=P)
    snes.setFunction(problem.F_nest, Fvec)
    snes.setJacobian(problem.J_nest, J=Jmat, P=Pmat)

    # snes.setDM(nullspace)

    

    x = fem.petsc.create_vector_nest(F)

    assert x.getType() == "nest"
    for x_soln_pair in zip(x.getNestSubVecs(), (u, p)):
        x_sub, soln_sub = x_soln_pair
        soln_sub.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        soln_sub.x.petsc_vec.copy(result=x_sub)
        x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Solve nonlinear problem
    snes.solve(None, x)
    assert snes.getKSP().getConvergedReason() > 0

    u.x.scatter_forward()
    p.x.scatter_forward()

    return u, p





