

from dolfinx.cpp.la.petsc import scatter_local_vectors
from dolfinx.fem import (Function, functionspace,
                         bcs_by_block, extract_function_spaces)

from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, la
from dolfinx_mpc  import (apply_lifting, assemble_matrix, 
                         assemble_matrix_nest, 
                            assemble_vector)
import dolfinx_mpc
from dolfinx.fem.petsc import set_bc, assemble_matrix_block, assemble_vector_block
# from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, 
#                                assemble_matrix_block, assemble_matrix_nest, 
#                                assemble_vector, assemble_vector_block, set_bc)







def linear_nested_solver(a, L, bcs, Pin, mpc_v, mpc_q):
    """Solve the Stokes problem using nest matrices and an iterative solver."""

    # Assemble nested matrix operators
    A = dolfinx_mpc.create_matrix_nest(a, [mpc_v, mpc_q])
    dolfinx_mpc.assemble_matrix_nest(A, a, [mpc_v, mpc_q], bcs)
    A.assemble()



    # Create a nested matrix P to use as the preconditioner. The
    # top-left block of P is shared with the top-left block of A. The
    # bottom-right diagonal entry is assembled from the form a_p11:
    P11 = fem.petsc.assemble_matrix(Pin)
    P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
    P.assemble()

    A00 = A.getNestSubMatrix(0, 0)
    A00.setOption(PETSc.Mat.Option.SPD, True)

    P00, P11 = P.getNestSubMatrix(0, 0), P.getNestSubMatrix(1, 1)
    P00.setOption(PETSc.Mat.Option.SPD, True)
    P11.setOption(PETSc.Mat.Option.SPD, True)

    # Assemble right-hand side vector
    b = dolfinx_mpc.create_vector_nest(L, [mpc_v, mpc_q])
    dolfinx_mpc.assemble_vector_nest(b, L, [mpc_v, mpc_q])

    # Modify ('lift') the RHS for Dirichlet boundary conditions
    fem.petsc.apply_lifting_nest(b, a, bcs=bcs)

    # Sum contributions for vector entries that are share across
    # parallel processes
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS vector
    bcs0 = fem.bcs_by_block(extract_function_spaces(L), bcs)
    fem.petsc.set_bc_nest(b, bcs0)

    # The pressure field is determined only up to a constant. We supply
    # a vector that spans the nullspace to the solver, and any component
    # of the solution in this direction will be eliminated during the
    # solution process.
    # null_vec = fem.petsc.create_vector_nest(L)

    # # Set velocity part to zero and the pressure part to a non-zero
    # # constant
    # null_vecs = null_vec.getNestSubVecs()
    # null_vecs[0].set(0.0), null_vecs[1].set(1.0)

    # # Normalize the vector that spans the nullspace, create a nullspace
    # # object, and attach it to the matrix
    # null_vec.normalize()
    # nsp = PETSc.NullSpace().create(vectors=[null_vec])
    # assert nsp.test(A)
    # A.setNullSpace(nsp)

    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setOperators(A, P)
    ksp.setType("minres")
    ksp.setTolerances(rtol=1e-8)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    # Define the matrix blocks in the preconditioner with the velocity
    # and pressure matrix index sets
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

    # Set the preconditioners for each block. For the top-left
    # Laplace-type operator we use algebraic multigrid. For the
    # lower-right block we use a Jacobi preconditioner. By default, GAMG
    # will infer the correct near-nullspace from the matrix block size.
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # Create finite element {py:class}`Function <dolfinx.fem.Function>`s
    # for the velocity (on the space `V`) and for the pressure (on the
    # space `Q`). The vectors for `u` and `p` are combined to form a
    # nested vector and the system is solved.
    x = b.copy()
    return ksp, x, b



