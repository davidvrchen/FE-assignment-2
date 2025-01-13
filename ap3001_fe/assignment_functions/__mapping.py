import numpy

def mapping_triangle(xi, eta, vertices):
    """Evaluates mapping between the canonical triangle element and a physical 
    triangle given by vertices.

    Computes the mapping (x, y) = Phi(xi, eta) that maps a point (xi, eta) in the 
    canonical triangle element

      (0,1) = V_3
            |\\
            |  \\ 
            |    \\
            |      \\
             -------
    (0,0) = V1     (1,0) = V_2   

    onto the physical triangle given by the three vertices
        U_1 = vertices[0, :]
        U_2 = vertices[1, :]
        U_3 = vertices[2, :]
    
    such that
        Phi(V_1) = U_1
        Phi(V_2) = U_2
        Phi(V_3) = U_3
        
    Returns the points (x_i, y_i) in the physical space
        (x_i, y_i) = Phi(xi_i, eta_i)
        
    Usage
    -----
        vertices = numpy.array([[1.0, -2.0], [3.0, 1.0], [2.0, 0.5]])
        xi = numpy.array([0.0, 1.0, 0.0])
        eta = numpy.array([0.0, 0.0, 1.0])
        x, y = ap3001_lib.mapping_triangle(xi, eta, vertices)

        matplotlib.pyplot.plot(x, y)
        matplotlib.pyplot.show()
    
    Parameters
    ----------
    xi : numpy.array(Float64), size (M, )
        The xi coordinates of the M points where to evaluate the mapping Phi.
    eta : numpy.array(Float64), size (M, )
        The eta coordinates of the M points where to evaluate the mapping Phi.
    vertices: numpy.array(Float64) size (3, 2)
        The x and y coordinates of the three vertices U_1, U_2, U_3 that define the triangle
        (in counterclockwise order).
                        -        -
            vertices = | x_1  y_1 |
                       | x_2  y_2 |
                       | x_3  y_3 |
                        -        - 
    
    Returns
    -------
    x : numpy.array(Float64), size (M,)
        The vector containing  x physical coordinates of the points inside the triangle
        that are the image of the canonical points (xi_i, eta_i) via the mapping Phi.
    y : numpy.array(Float64), size (M,)
        The vector containing the y physical coordinates of the points inside the triangle
        that are the image of the canonical points (xi_i, eta_i) via the mapping Phi.
    """

    # Check if inputs have the correct shape
    if (len(xi.shape) > 1) or (len(eta.shape) > 1):
        raise Exception("The input coordinates must be vectors, not arrays.")
    
    if (len(xi) != len(eta)):
        raise Exception("The input coordinates must have the same number of coordinates.")

    x = vertices[0][0] + xi*(vertices[1][0] - vertices[0][0]) + eta*(vertices[2][0] - vertices[0][0])
    y = vertices[0][1] + xi*(vertices[1][1] - vertices[0][1]) + eta*(vertices[2][1] - vertices[0][1])

    return x, y
    

def jacobian_triangle(xi, eta, vertices):
    """Evaluates Jacobian matrix of the mapping between the canonical 
    triangle element and a physical triangle given by vertices.

    Computes the Jacobian matrix
             -                       -
        J = | dPhi^x/dxi  dPhi^x/deta |
            | dPhi^y/dxi  dPhi^y/deta |
             -                       -

    of the mapping (x, y) = Phi(xi, eta) that maps a point (xi, eta) in the 

    canonical triangle element

      (0,1) = V_3
            |\\
            |  \\ 
            |    \\
            |      \\
             -------
    (0,0) = V1     (1,0) = V_2   

    onto the physical triangle given by the three vertices
        U_1 = vertices[0, :]
        U_2 = vertices[1, :]
        U_3 = vertices[2, :]
    
    such that
        Phi(V_1) = U_1
        Phi(V_2) = U_2
        Phi(V_3) = U_3
        
    Returns the Jacobian matrix evaluated at the points (xi_i, eta_i)
        J[i, :, :]

    i.e.
        J[1, 1, 1] =  dPhi^y/deta (xi_2, eta_2) (note that the nodes are indexed base 1 and Python uses base 0)
        J[1, 0, 1] =  dPhi^x/deta (xi_2, eta_2)
        J[k, 1, 0] =  dPhi^y/dxi (xi_k, eta_k)

                      -                       -
        J[k, :, :] = | dPhi^x/dxi  dPhi^x/deta | (xi_k, eta_k) (i.e., evaluated at point (xi_k, eta_k)
                     | dPhi^y/dxi  dPhi^y/deta |
                      -                       -
        
    Usage
    -----
        vertices = numpy.array([[1.0, -2.0], [3.0, 1.0], [2.0, 0.5]])
        xi = numpy.array([0.0, 1.0, 0.0])
        eta = numpy.array([0.0, 0.0, 1.0])
        J = ap3001_lib.jacobian_triangle(xi, eta, vertices)
    
    Parameters
    ----------
    xi : numpy.array(Float64), size (M, )
        The xi coordinates of the M points where to evaluate the mapping Phi.
    eta : numpy.array(Float64), size (M, )
        The eta coordinates of the M points where to evaluate the mapping Phi.
    vertices: numpy.array(Float64) size (3, 2)
        The x and y coordinates of the three vertices U_1, U_2, U_3 that define the triangle
        (in counterclockwise order).
                        -        -
            vertices = | x_1  y_1 |
                       | x_2  y_2 |
                       | x_3  y_3 |
                        -        - 
    
    Returns
    -------
    J : numpy.array(Float64), size (M, 2, 2)
        The Jacobian matrix evaluated at the points (xi_i, eta_i)
            J[i, :, :]
        i.e.
            J[1, 1, 1] =  dPhi^y/deta (xi_2, eta_2) (note that the nodes are indexed base 1 and Python uses base 0)
            J[1, 0, 1] =  dPhi^x/deta (xi_2, eta_2)
            J[k, 1, 0] =  dPhi^y/dxi (xi_k, eta_k)

                          -                       -
            J[k, :, :] = | dPhi^x/dxi  dPhi^x/deta | (xi_k, eta_k) (i.e., evaluated at point (xi_k, eta_k)
                         | dPhi^y/dxi  dPhi^y/deta |
                          -                       -
    """

    # Check if inputs have the correct shape
    if (len(xi.shape) > 1) or (len(eta.shape) > 1):
        raise Exception("The input coordinates must be vectors, not arrays.")
    
    if (len(xi) != len(eta)):
        raise Exception("The input coordinates must have the same number of coordinates.")
    
    # Check the number of points where to evaluate the basis
    n_eval_points = len(xi)  # we can now evaluate just one of the inputs since at this stage
                             # we are sure they have the same shape

    # Preallocate memory for the Jacobian output
    physical_manifold_dimension = 2  # the physical triangle is embedded in 2D space
    canonical_manifold_dimension = 2  # the canonical triangle is also in 2D space
    J = numpy.zeros([n_eval_points, physical_manifold_dimension, canonical_manifold_dimension])

    # Compute each term of the Jacobian (note that it is the same for all points)
    dPhi_x_d_xi =   # TODO
    dPhi_x_d_eta =  # TODO
    dPhi_y_d_xi =   # TODO
    dPhi_y_d_eta =  # TODO

    # Then place it in the Jacobian matrix
    # We could have made this into a 2D matrix instead of repeating the same for all points
    # This less efficient repetition is done so that you are aware that, in general, 
    # the Jacobian does not have to be the same for all points.
    # For triangles mapped by an affine transformation, the Jacobian is constant inside each 
    # element.
    J[:, 0, 0] = dPhi_x_d_xi
    J[:, 0, 1] = dPhi_x_d_eta
    J[:, 1, 0] = dPhi_y_d_xi
    J[:, 1, 1] = dPhi_y_d_eta

    return J 