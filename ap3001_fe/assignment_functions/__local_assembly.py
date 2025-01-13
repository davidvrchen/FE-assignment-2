import numpy

def generate_element_mass_matrix(basis, element_idx, sigma):
    # Get the quadrature weights and nodes
    xi_quad, eta_quad, w_quad = basis.quadrature()

    # Compute the canonical basis
    # We need to integrate the inner products of canonical basis
    #   \int B_c_{i}(\xi, \eta) B_c_{j}(\xi, \eta) |det J(\xi, \eta)| d\Omega^c
    # where \Omega^c is the canonical element.

    # To do that we use quadrature, therefore 
    # we need to evaluate the canonical basis on the quadrature nodes.
    Bc_basis_eval = basis.basis_canonical(xi_quad, eta_quad)

    # Compute the Jacobian determinant
    # In lectures we also saw that this was the square root of the 
    # determinant of the inverse of the metric matrix G = (J^T) J
    # i.e., det J = sqrt(det(inv(G)))
    det_J = basis.mesh.sqrt_det_jacobian_transpose_times_jacobian(xi_quad, eta_quad, element_idx)

    # Compute the integrals
    #   \int B_c_{i}(\xi, \eta) B_c_{j}(\xi, \eta) |det J(\xi, \eta)| d\Omega^c
    # using quadrature, where \Omega^c is the canonical element.
    n_canonical_basis = basis.n_canonical_basis  # get the number of canonical basis in the element
    M_local = numpy.zeros([n_canonical_basis, n_canonical_basis])  # pre-allocate memory for the local mass matrix

    for i_basis_idx in numpy.arange(0, 3):
        for j_basis_idx in numpy.arange(0, 3):
            M_local[i_basis_idx, j_basis_idx] = w_quad @ (Bc_basis_eval[i_basis_idx, :] * Bc_basis_eval[j_basis_idx, :] * det_J)

    return M_local


def generate_element_stiffness_matrix(basis, element_idx, sigma):
    # Get the quadrature weights and nodes
    xi_quad, eta_quad, w_quad = basis.quadrature()
    n_quad_nodes = len(xi_quad)  # the number of quadrature nodes

    # Compute the canonical basis
    # We need to integrate the inner products of canonical basis
    #   \int sigma \nabla B_c_{i}(\xi, \eta) G(\xi, \eta)^-1 \nabla B_c_{j}(\xi, \eta) |det J(\xi, \eta)| d\Omega^c
    # where \Omega^c is the canonical element.

    # To do that we use quadrature, therefore 
    # we need to evaluate the canonical basis on the quadrature nodes.
    grad_Bc_basis_eval = basis.grad_basis_canonical(xi_quad, eta_quad)

    # Compute the Jacobian determinant
    # In the lectures we also saw that this was the square root of the 
    # determinant of the inverse of the metric matrix G = (J^T) J
    # i.e., det J = sqrt(det(inv(G)))
    det_J = basis.mesh.sqrt_det_jacobian_transpose_times_jacobian(xi_quad, eta_quad, element_idx)

    # Compute the inverse of the metric matrix (actually it is a tensor) G^-1
    # At the quadrature nodes, since this is what we need to compute quadrature of
    #   \int \nabla B_c_{i}(\xi, \eta) G(\xi, \eta)^-1 \nabla B_c_{j}(\xi, \eta) |det J(\xi, \eta)| d\Omega^c
    # where \Omega^c is the canonical element.
    inv_G = basis.mesh.inv_jacobian_transpose_times_jacobian(xi_quad, eta_quad, element_idx)


    # Compute the integrals
    #   \int sigma \nabla B_c_{i}(\xi, \eta) G(\xi, \eta)^-1 \nabla B_c_{j}(\xi, \eta) |det J(\xi, \eta)| d\Omega^c
    # using quadrature
    n_canonical_basis = basis.n_canonical_basis  # get the number of canonical basis in the element
    N_local = numpy.zeros([n_canonical_basis, n_canonical_basis])  # pre-allocate memory for the local mass matrix

    for i_basis_idx in numpy.arange(0, 3):
        for j_basis_idx in numpy.arange(0, 3):
            for quad_node_idx in numpy.arange(0, n_quad_nodes):
                N_local[i_basis_idx, j_basis_idx] += sigma*w_quad[quad_node_idx] * grad_Bc_basis_eval[i_basis_idx, quad_node_idx, :] @ (inv_G[quad_node_idx, :, :] @ grad_Bc_basis_eval[j_basis_idx, quad_node_idx, :]) * det_J[quad_node_idx]

    return N_local




def generate_element_vector(basis, element_idx, f):
    # Get the quadrature weights and nodes
    xi_quad, eta_quad, w_quad = basis.quadrature()
    n_quad_nodes = len(xi_quad)  # the number of quadrature nodes

    # Get the point in physical space (x, y) corresponding to the quadrature nodes
    # We will need to evaluate f at those points, for the quadrature 
    x_quad, y_quad = basis.mesh.mapping(xi_quad, eta_quad, element_idx)

    # Compute the canonical basis
    # We need to integrate the inner products of canonical basis
    #   \int f(\Phi_{l}(xi, eta)) B_c_{j}(\xi, \eta) |det J| d\xi d\eta
    # To do that we use quadrature, therefore 
    # we need to evaluate the canonical basis on the quadrature nodes.
    Bc_basis_eval = basis.basis_canonical(xi_quad, eta_quad)

    # Compute the Jacobian determinant
    # In the lectures we also saw that this was the square root of the 
    # determinant of the inverse of the metric matrix G = (J^T) J
    # i.e., det J = sqrt(det(inv(G)))
    det_J = basis.mesh.sqrt_det_jacobian_transpose_times_jacobian(xi_quad, eta_quad, element_idx)

    # Compute the integrals
    #   \int f(\Phi_{l}(xi, eta)) B_c_{j}(\xi, \eta) |det J| d\Omega_l
    # using quadrature
    n_canonical_basis = basis.n_canonical_basis  # get the number of canonical basis in the element
    F_local = numpy.zeros(n_canonical_basis)  # pre-allocate memory for the local vector

    for j_basis_idx in numpy.arange(0, 3):
        F_local[j_basis_idx] = w_quad @ (f(x_quad, y_quad) * Bc_basis_eval[j_basis_idx, :] * det_J)

    return F_local








