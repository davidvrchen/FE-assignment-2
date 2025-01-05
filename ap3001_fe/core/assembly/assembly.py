
import numpy
from scipy.sparse import lil_matrix as sparse_matrix

from ...assignment_functions import *


# Boundary conditions ---------------------------------------------------------

def impose_boundary_conditions_matrix(M, basis):
    for essential_bc_key in basis.essential_bcs:
        for boundary_basis_idx, boundary_value in zip(basis.essential_bcs[essential_bc_key]['indices'], basis.essential_bcs[essential_bc_key]['values']):
            # Modify the matrix M to enforce the boundary condition
            M[boundary_basis_idx,:] = 0.0
            M[boundary_basis_idx, boundary_basis_idx] = 1.0

    return M

def impose_boundary_conditions_vector(f, basis):
    for essential_bc_key in basis.essential_bcs:
        for boundary_basis_idx, boundary_value in zip(basis.essential_bcs[essential_bc_key]['indices'], basis.essential_bcs[essential_bc_key]['values']):
            # Modify the vector to assign the boundary condition value
            f[boundary_basis_idx] = boundary_value

    return f

# -----------------------------------------------------------------------------



# Assembly --------------------------------------------------------------------

def assemble_global_mass_matrix(basis, sigma):
    # We need to compute the mass matrix M, with elements
    #   M_{i,s} = \int_{\Omega} \varphi_{i} \varphi_{r} dx dy
    # As we saw in the lectures, we can break this down into 
    # a sum of the contributions from each of the K elements \Omega_{l}
    # l = 1, ..., M 
    #
    #   M_{i,r} = \sum_{l=1}^{K}\int_{\Omega_{l}} \varphi_{i} \varphi_{r} dx dy
    #
    # We also saw that each basis \varphi_{i} can be expressed as a linear 
    # combination of the local basis B_l_j
    # 
    #   \varphi_{i}(x, y) = \sum_{l=1}^{K}\sum_{j=1}^{n_local_basis} E_{l,i,j} B_{l, j}(x, y) 
    #
    #  For this reason, we have
    # 
    #   M_{i,r} = \sum_{l=1}^{K}\sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s}\int_{\Omega_{l}} B_{l,j}(x, y) B_{l, s}(x, y) dx dy
    # 
    # The integral of the local basis, could be done via Gauss quadrature. To use
    # Gauss quadrature, we first need to convert the integral from the element \Omega_{l}
    # to the canonical element \Omega_{c}, as we saw in the lectures 
    #
    #   M_{i,r} = \sum_{l=1}^{K}\sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s} \int_{\Omega_{c}} B_{l,j}(\Phi_{l}(xi, eta)) B_{l, s}(\Phi_{l}(xi, eta)) |det(J(xi, eta))| dxi deta
    #
    # If we recall that B_{l, s}(x, y) = Bc_s(\Phi^{-1}_{l}(x, y)), then 
    #
    #   B_{l, s}(\Phi_{l}(xi, eta)) = Bc_s(\Phi^{-1}_{l}(\Phi_{l}(xi, eta))
    #                               = Bc_s(xi, eta)
    #
    # and we can write 
    # 
    #   M_{i,r} = \sum_{l=1}^{K}\sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s} \int_{\Omega_{c}} Bc_{j}(xi, eta) Bc_{s}(xi, eta) |det(J(xi, eta))| dxi deta
    #
    # which, if you recall, is just 
    #
    #    M_{i,r} = \sum_{l=1}^{K}\sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s} M_local_{js}

    # Get the number of elements 
    n_elements = basis.mesh.number_of_elements()

    # Get the number of local/canonical basis
    n_canonical_basis = basis.n_canonical_basis

    # Determine the size of the matrix 
    # NOTE: Here we assemble the whole mass matrix, i.e., we compute all possible inner products 
    #       without ignoring the boundary basis. This is more efficient. We will then deal with 
    #       the essential boundary conditions in another function. For now, the focus is to contruct 
    #       a mass matrix with the inner products between all basis.
    n_basis = basis.n_basis  # the total number of basis \varphi_{j}(x, y)

    # Setup the global matrix to store the data 
    M_global = sparse_matrix((n_basis, n_basis), dtype='float64')  # it is a sparse matrix, so that we only need to store 
                                                                   # the nonzero elements 

    # Loop over the elements of the mesh and add the contribution from that element, i.e.,
    # as we pass over each element we will add 
    #
    #   \sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s} M_local_{js}
    #
    # once we pass by all elements, we will have added all contributions and we will have 
    #
    #   M_{i,r} = \sum_{l=1}^{K}\sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s} M_local_{js}
    for element_idx in numpy.arange(0, n_elements):
        # Compute the local mass matrix M_local 
        M_local = generate_element_mass_matrix(basis, element_idx, sigma)

        # Add the contribution to the global matrix 
        # We could loop over all basis (even the ones that we know are zero on the element)
        # but this would be very slow. That approach is given below (commented out). 

        # for basis_i_idx in numpy.arange(n_basis):
        #     for basis_r_idx in numpy.arange(n_basis):
        #         for canonical_basis_j_idx in numpy.arange(0, n_canonical_basis):
        #             for canonical_basis_s_idx in numpy.arange(0, n_canonical_basis):
        #                 M_global[basis_i_idx, basis_r_idx] += basis.E_active_basis[element_idx, basis_i_idx, canonical_basis_j_idx] * basis.E_active_basis[element_idx, basis_r_idx, canonical_basis_s_idx] * M_local[canonical_basis_j_idx, canonical_basis_s_idx]

        # An alternative is to loop only over the active basis on each element, i.e., the basis 
        # that are nonzero in the element. For that, instead of using the full extraction 
        # coefficient matrix
        #
        #    basis.E_full
        #
        # we use the efficient version of it, that stores only the coefficients for the active basis 
        #
        #   basis.E_active_basis
        #
        # and the indices of the active basis
        #
        #   basis.element_active_basis_idx
        #
        # In the questions, you were asked to show that the two approaches are equivalent. Implement 
        # this efficient approach here
        for active_basis_i_local_idx, active_basis_i_global_idx in enumerate(basis.element_active_basis_idx[element_idx]):
            for active_basis_r_local_idx, active_basis_r_global_idx in enumerate(basis.element_active_basis_idx[element_idx]):
                for canonical_basis_j_idx in numpy.arange(0, n_canonical_basis):
                    for canonical_basis_s_idx in numpy.arange(0, n_canonical_basis):
                        M_global[active_basis_i_global_idx, active_basis_r_global_idx] += basis.E_active_basis[element_idx, active_basis_i_local_idx, canonical_basis_j_idx] * basis.E_active_basis[element_idx, active_basis_r_local_idx, canonical_basis_s_idx] * M_local[canonical_basis_j_idx, canonical_basis_s_idx]

    return M_global


def assemble_global_stiffness_matrix(basis, sigma):
    # We need to compute the stiffness matrix N, with elements
    #   N_{i,s} = \int_{\Omega} nabla\varphi_{i} \cdot \nabla\varphi_{r} dx dy
    # As we saw in the lectures, we can break this down into 
    # a sum of the contributions from each of the K elements \Omega_{l}
    # l = 1, ..., M 
    #
    #   N_{i,r} = \sum_{l=1}^{K}\int_{\Omega_{l}} \nabla\varphi_{i} \cdot \nabla\varphi_{r} dx dy
    #
    # We also saw that each basis \varphi_{i} can be expressed as a linear 
    # combination of the local basis B_l_j
    # 
    #   \varphi_{i}(x, y) = \sum_{l=1}^{K}\sum_{j=1}^{n_local_basis} E_{l,i,j} B_{l, j}(x, y) 
    #
    #  For this reason, we have
    # 
    #   N_{i,r} = \sum_{l=1}^{K}\sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s}\int_{\Omega_{l}} \nabla B_{l,j}(x, y) \cdot \nabla B_{l, s}(x, y) dx dy
    # 
    # The integral of the local basis, could be done via Gauss quadrature. To use
    # Gauss quadrature, we first need to convert the integral from the element \Omega_{l}
    # to the canonical element \Omega_{c}, as we saw in the lectures 
    #
    #   N_{i,r} = \sum_{l=1}^{K}\sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s} \int_{\Omega_{c}} \nabla^{c} B_{l,j}(\Phi_{l}(xi, eta)) (J^{T} J)^{-1} \nabla^{c} B_{l, s}(\Phi_{l}(xi, eta)) |det(J(xi, eta))| dxi deta
    #
    # If we recall that B_{l, s}(x, y) = Bc_s(\Phi^{-1}_{l}(x, y)), then 
    #
    #   B_{l, s}(\Phi_{l}(xi, eta)) = Bc_s(\Phi^{-1}_{l}(\Phi_{l}(xi, eta))
    #                               = Bc_s(xi, eta)
    #
    # and we can write 
    # 
    #   N_{i,r} = \sum_{l=1}^{K}\sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s} \int_{\Omega_{c}} \nabla^{c} Bc_{j}(xi, eta) (J^{T} J)^{-1} \nabla^{c} Bc_{s}(xi, eta) |det(J(xi, eta))| dxi deta
    #
    # which, if you recall, is just 
    #
    #    N_{i,r} = \sum_{l=1}^{K}\sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s} N_local_{js}

    # The assembly procedure, as expected, is exactly the same as for the mass matrix M.

    # Get the number of elements 
    n_elements = basis.mesh.number_of_elements()

    # Get the number of local/canonical basis
    n_canonical_basis = basis.n_canonical_basis

    # Determine the size of the matrix 
    # NOTE: Here we assemble the whole mass matrix, i.e., we compute all possible inner products 
    #       without ignoring the boundary basis. This is more efficient. We will then deal with 
    #       the essential boundary conditions in another function. For now, the focus is to contruct 
    #       a mass matrix with the inner products between all basis.
    n_basis = basis.n_basis  # the total number of basis \varphi_{j}(x, y)

    # Setup the global matrix to store the data 
    N_global = sparse_matrix((n_basis, n_basis), dtype='float64')  # it is a sparse matrix, so that we only need to store 
                                                                   # the nonzero elements 

    # Loop over the elements of the mesh and add the contribution from that element, i.e.,
    # as we pass over each element we will add 
    #
    #   \sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s} N_local_{js}
    #
    # once we pass by all elements, we will have added all contributions and we will have 
    #
    #   N_{i,r} = \sum_{l=1}^{K}\sum_{j,s = 1}^{n_local_basis} E_{l,i,j} E_{l,r,s} N_local_{js}
    for element_idx in numpy.arange(0, n_elements):
        # Compute the local mass matrix N_local 
        N_local = generate_element_stiffness_matrix(basis, element_idx, sigma)

        # Add the contribution to the global matrix 
        # We could loop over all basis (even the ones that we know are zero on the element)
        # but this would be very slow. That approach is given below (commented out). 

        # for basis_i_idx in numpy.arange(n_basis):
        #     for basis_r_idx in numpy.arange(n_basis):
        #         for canonical_basis_j_idx in numpy.arange(0, n_canonical_basis):
        #             for canonical_basis_s_idx in numpy.arange(0, n_canonical_basis):
        #                 N_global[basis_i_idx, basis_r_idx] += basis.E_active_basis[element_idx, basis_i_idx, canonical_basis_j_idx] * basis.E_active_basis[element_idx, basis_r_idx, canonical_basis_s_idx] * N_local[canonical_basis_j_idx, canonical_basis_s_idx]

        # An alternative is to loop only over the active basis on each element, i.e., the basis 
        # that are nonzero in the element. For that, instead of using the full extraction 
        # coefficient matrix
        #
        #    basis.E_full
        #
        # we use the efficient version of it, that stores only the coefficients for the active basis 
        #
        #   basis.E_active_basis
        #
        # and the indices of the active basis
        #
        #   basis.element_active_basis_idx
        #
        # In the questions, you were asked to show that the two approaches are equivalent. Implement 
        # this efficient approach here
        for active_basis_i_local_idx, active_basis_i_global_idx in enumerate(basis.element_active_basis_idx[element_idx]):
            for active_basis_r_local_idx, active_basis_r_global_idx in enumerate(basis.element_active_basis_idx[element_idx]):
                for canonical_basis_j_idx in numpy.arange(0, n_canonical_basis):
                    for canonical_basis_s_idx in numpy.arange(0, n_canonical_basis):
                        N_global[active_basis_i_global_idx, active_basis_r_global_idx] += basis.E_active_basis[element_idx, active_basis_i_local_idx, canonical_basis_j_idx] * basis.E_active_basis[element_idx, active_basis_r_local_idx, canonical_basis_s_idx] * N_local[canonical_basis_j_idx, canonical_basis_s_idx]

    return N_global


def assemble_global_vector(basis, f):
    # We need to compute the right hand side vector f, with elements
    #   f_{j} = \int_{\Omega} f(x, y) \varphi_{j} dx dy
    # As we saw in the lectures, we can break this down into 
    # a sum of the contributions from each of the K elements \Omega_{l}
    # l = 1, ..., M 
    #
    #   f_{r} = \sum_{l=1}^{K}\int_{\Omega_{l}} f(x, y) \varphi_{r} dx dy
    #
    # We also saw that each basis \varphi_{r} can be expressed as a linear 
    # combination of the local basis B_l_s
    # 
    #   \varphi_{r}(x, y) = \sum_{l=1}^{K}\sum_{s=1}^{n_local_basis} E_{l,r,s} B_{l, s}(x, y) 
    #
    #  For this reason, we have
    # 
    #   f_{r} = \sum_{l=1}^{K}\sum_{s = 1}^{n_local_basis} E_{l,r,s}\int_{\Omega_{l}} f(x, y) B_{l, s}(x, y) dx dy
    # 
    # The integral could be done via Gauss quadrature. To use
    # Gauss quadrature, we first need to convert the integral from the element \Omega_{l}
    # to the canonical element \Omega_{c}, as we saw in the lectures 
    #
    #   f_{r} = \sum_{l=1}^{K}\sum_{s = 1}^{n_local_basis} E_{l,r,s} \int_{\Omega_{c}} f(\Phi(xi, eta)) B_{l, s}(\Phi_{l}(xi, eta)) |det(J(xi, eta))| dxi deta
    #
    # If we recall that B_{l, s}(x, y) = Bc_s(\Phi^{-1}_{l}(x, y)), then 
    #
    #   B_{l, s}(\Phi_{l}(xi, eta)) = Bc_s(\Phi^{-1}_{l}(\Phi_{l}(xi, eta))
    #                               = Bc_s(xi, eta)
    #
    # and we can write 
    # 
    #   f_{r} = \sum_{l=1}^{K}\sum_{s = 1}^{n_local_basis} E_{l,r,s} \int_{\Omega_{c}} f(\Phi(xi, eta)) Bc_{s}(xi, eta) |det(J(xi, eta))| dxi deta
    #
    # which, if you recall, is just 
    #
    #    f_{r} = \sum_{l=1}^{K}\sum_{s = 1}^{n_local_basis} E_{l,r,s} f_local_{s}

    # The assembly procedure, as expected, is exactly the same as for the mass and stiffness 
    # matrices, just that now we only have to loop over one set of global basi \varphi_{s},
    # instead of two.

    # Get the number of elements 
    n_elements = basis.mesh.number_of_elements()

    # Get the number of local/canonical basis
    n_canonical_basis = basis.n_canonical_basis

    # Determine the size of the matrix 
    # NOTE: Here we assemble the whole vector, i.e., we compute all possible inner products 
    #       without ignoring the boundary basis. This is more efficient. We will then deal with 
    #       the essential boundary conditions in another function. For now, the focus is to contruct 
    #       a vector with all basis.
    n_basis = basis.n_basis  # the total number of basis \varphi_{s}(x, y)

    # Setup the global vector to store the data 
    f_global = numpy.zeros(n_basis)

    # Loop over the elements of the mesh and add the contribution from that element, i.e.,
    # as we pass over each element we will add 
    #
    #   \sum_{j,s = 1}^{n_local_basis} E_{l,r,s} f_local_{s}
    #
    # once we pass by all elements, we will have added all contributions and we will have 
    #
    #   f_{r} = \sum_{l=1}^{K}\sum_{s = 1}^{n_local_basis} E_{l,r,s} f_local_{s}
    for element_idx in numpy.arange(0, n_elements):
        # Compute the local vector f_local 
        f_local = generate_element_vector(basis, element_idx, f)

        # Add the contribution to the global vector 
        # We could loop over all basis (even the ones that we know are zero on the element)
        # but this would be very slow. That approach is given below (commented out). 

        # for basis_r_idx in numpy.arange(n_basis):
        #     for canonical_basis_s_idx in numpy.arange(0, n_canonical_basis):
        #         f_global[basis_r_idx] += basis.E_active_basis[element_idx, basis_r_idx, canonical_basis_s_idx] * f_local[canonical_basis_s_idx]

        # An alternative is to loop only over the active basis on each element, i.e., the basis 
        # that are nonzero in the element. For that, instead of using the full extraction 
        # coefficient matrix
        #
        #    basis.E_full
        #
        # we use the efficient version of it, that stores only the coefficients for the active basis 
        #
        #   basis.E_active_basis
        #
        # and the indices of the active basis
        #
        #   basis.element_active_basis_idx
        #
        # In the questions, you were asked to show that the two approaches are equivalent. Implement 
        # this efficient approach here
        for active_basis_r_local_idx, active_basis_r_global_idx in enumerate(basis.element_active_basis_idx[element_idx]):
            for canonical_basis_s_idx in numpy.arange(0, n_canonical_basis):
                f_global[active_basis_r_global_idx] += basis.E_active_basis[element_idx, active_basis_r_local_idx, canonical_basis_s_idx] * f_local[canonical_basis_s_idx]

    return f_global

# -----------------------------------------------------------------------------