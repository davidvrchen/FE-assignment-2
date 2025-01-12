import numpy


def basis_canonical_triangle(xi, eta):
    """Evaluates the canonical basis that enable the construction of hat functions.

    The three canonical basis, B_c_1, B_c_2, and B_c_3 on the canonical triangle

      (0,1) = V_3
            |\\
            |  \\ 
            |    \\
            |      \\
             -------
    (0,0) = V1     (1,0) = V_2   

    are linear functions given by
        
        B_c_1(xi, eta) = alpha_1 + beta_1 xi + gamma_1 eta
        B_c_2(xi, eta) = alpha_2 + beta_2 xi + gamma_2 eta
        B_c_2(xi, eta) = alpha_3 + beta_3 xi + gamma_3 eta

    with the property that
        B_c_1(V_1) = 1.0, B_c_1(V_2) = 0.0, and B_c_1(V_3) = 0.0
        B_c_2(V_1) = 0.0, B_c_2(V_2) = 1.0, and B_c_2(V_3) = 0.0
        B_c_3(V_1) = 0.0, B_c_3(V_2) = 0.0, and B_c_3(V_3) = 1.0 
    
    Returns the 3 canonical basis evaluated at the M input points 
        P_1 = (xi_1, eta_1)
           ...
        P_M = (xi_M, eta_M) 
    
    as a 3 x M matrix:
                    -                         -
        B_c_eval = | B_c_1(P_1) ... B_c_1(P_M) |
                   | B_c_2(P_1) ... B_c_2(P_M) |
                   | B_c_3(P_1) ... B_c_3(P_M) |
                    -                         -
    Usage
    -----
        xi = numpy.array([0.0, 1.0, 0.0])
        eta = numpy.array([0.0, 0.0, 1.0])
        B_c_eval = basis_canonical_triangle(xi, eta)
    
    Parameters
    ----------
    xi : numpy.array(Float64), size (M, )
        The xi coordinates of the M points where to evaluate the three canonical basis.
    eta : numpy.array(Float64), size (M, )
        The eta coordinates of the M points where to evaluate the three canonical basis.
    
    Returns
    -------
    B_c_eval : numpy.array(Float64), size (3, M)
        The array containing the evaluation of the three canonical basis on the M nodes (xi_i, eta_i).
    """

    # Check if inputs have the correct shape
    if (len(xi.shape) > 1) or (len(eta.shape) > 1):
        raise Exception("The input coordinates must be vectors, not arrays.")
    
    if (len(xi) != len(eta)):
        raise Exception("The input coordinates must have the same number of coordinates.")
    
    # Check the number of points where to evaluate the basis
    n_eval_points = len(xi)  # we can now evaluate just one of the inputs since at this stage
                             # we are sure they have the same shape

    # Pre-allocate results
    n_basis = 3  # we have three local basis for hat functions on a triangular element
    B_c_eval = numpy.zeros([n_basis, n_eval_points])
    
    # Basis associated to node 1 (xi, eta) = (0.0, 0.0)
    alpha_1 = 1
    beta_1 =  -1
    gamma_1 = -1

    B_c_eval[0, :] = alpha_1 + beta_1 * xi + gamma_1 * eta 

    # Basis associated to node 2 (xi, eta) = (1.0, 0.0)
    alpha_2 = 0
    beta_2 =  1
    gamma_2 = 0

    B_c_eval[1, :] = alpha_2 + beta_2 * xi + gamma_2 * eta 

    # Basis associated to node 3 (xi, eta) = (0.0, 1.0)
    alpha_3 =  0
    beta_3 =   0
    gamma_3 =  1

    B_c_eval[2, :] = alpha_3 + beta_3 * xi + gamma_3 * eta 

    return B_c_eval


def grad_basis_canonical_triangle(xi, eta):
    """Evaluates the gradient of canonical basis that enable the construction of hat functions.

    The three canonical basis, B_c_1, B_c_2, and B_c_3 on the canonical triangle

      (0,1) = V_3
            |\\
            |  \\ 
            |    \\
            |      \\
             -------
    (0,0) = V1     (1,0) = V_2   

    are linear functions given by
        
        B_c_1(xi, eta) = alpha_1 + beta_1 xi + gamma_1 eta
        B_c_2(xi, eta) = alpha_2 + beta_2 xi + gamma_2 eta
        B_c_2(xi, eta) = alpha_3 + beta_3 xi + gamma_3 eta

    with the property that
        B_c_1(V_1) = 1.0, B_c_1(V_2) = 0.0, and B_c_1(V_3) = 0.0
        B_c_2(V_1) = 0.0, B_c_2(V_2) = 1.0, and B_c_2(V_3) = 0.0
        B_c_3(V_1) = 0.0, B_c_3(V_2) = 0.0, and B_c_3(V_3) = 1.0 
    
    Returns the gradient of the 3 canonical basis evaluated at the M input points 
        P_1 = (xi_1, eta_1)
           ...
        P_M = (xi_M, eta_M) 
    
    as a 3 x M x 2 matrix:
                                  -                                   -
        grad_B_c_eval[:, :, 0] = | dB_c_1_dxi(P_1) ... dB_c_1_dxi(P_M) |
                                 | dB_c_2_dxi(P_1) ... dB_c_2_dxi(P_M) |
                                 | dB_c_3_dxi(P_1) ... dB_c_3_dxi(P_M) |
                                  -                                   -
                                  -                                   -
        grad_B_c_eval[:, :, 1] = | dB_c_1_deta(P_1) ... dB_c_1_deta(P_M) |
                                 | dB_c_2_deta(P_1) ... dB_c_2_deta(P_M) |
                                 | dB_c_3_deta(P_1) ... dB_c_3_deta(P_M) |
                                  -                                   -
    Usage
    -----
        xi = numpy.array([0.0, 1.0, 0.0])
        eta = numpy.array([0.0, 0.0, 1.0])
        grad_B_c_eval = grad_basis_canonical_triangle(xi, eta)
    
    Parameters
    ----------
    xi : numpy.array(Float64), size (M, )
        The xi coordinates of the M points where to evaluate the three canonical basis.
    eta : numpy.array(Float64), size (M, )
        The eta coordinates of the M points where to evaluate the three canonical basis.
    
    Returns
    -------
    grad_B_c_eval : numpy.array(Float64), size (3, M, 2)
        The array containing the evaluation of the gradient of the
        three canonical basis on the M nodes (xi_i, eta_i).
                                      -                                   -
            grad_B_c_eval[:, :, 0] = | dB_c_1_dxi(P_1) ... dB_c_1_dxi(P_M) |
                                     | dB_c_2_dxi(P_1) ... dB_c_2_dxi(P_M) |
                                     | dB_c_3_dxi(P_1) ... dB_c_3_dxi(P_M) |
                                      -                                   -

                                       -                                   -
            grad_ B_c_eval[:, :, 1] = | dB_c_1_deta(P_1) ... dB_c_1_deta(P_M) |
                                      | dB_c_2_deta(P_1) ... dB_c_2_deta(P_M) |
                                      | dB_c_3_deta(P_1) ... dB_c_3_deta(P_M) |
                                       -                                   -
    """

    # Check if inputs have the correct shape
    if (len(xi.shape) > 1) or (len(eta.shape) > 1):
        raise Exception("The input coordinates must be vectors, not arrays.")
    
    if (len(xi) != len(eta)):
        raise Exception("The input coordinates must have the same number of coordinates.")
    
    # Check the number of points where to evaluate the basis
    n_eval_points = len(xi)  # we can now evaluate just one of the inputs since at this stage
                             # we are sure they have the same shape

    # Pre-allocate results
    n_basis = 3  # we have three local basis for hat functions on a triangular element
    grad_B_c_eval = numpy.zeros([n_basis, n_eval_points, 2])
    
    # Basis associated to node 1 (xi, eta) = (0.0, 0.0)
    alpha_1 = # TODO
    beta_1 =  # TODO
    gamma_1 = # TODO

    grad_B_c_eval[0, :, 0] =  # TODO 
    grad_B_c_eval[0, :, 1] =  # TODO 

    # Basis associated to node 2 (xi, eta) = (1.0, 0.0)
    alpha_2 = # TODO
    beta_2 =  # TODO
    gamma_2 = # TODO

    grad_B_c_eval[1, :, 0] = # TODO 
    grad_B_c_eval[1, :, 1] = # TODO 

    # Basis associated to node 3 (xi, eta) = (0.0, 1.0)
    alpha_3 =  # TODO
    beta_3 =   # TODO
    gamma_3 =  # TODO

    grad_B_c_eval[2, :, 0] = # TODO 
    grad_B_c_eval[2, :, 1] = # TODO 

    return grad_B_c_eval