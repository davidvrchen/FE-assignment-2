import numpy


def gauss_quadrature_2D_triangle(N):
    """Returns the Gauss quadrature weights and nodes for 2D quadrature of degree N over triangles.

    Returns the M 2D Gauss quadrature weights associated to quadrature of degree N
        w_{i}, i = 1, ..., M
    associated to the nodes    
        (xi, eta)_{i} in the canonical triangle (Omega_t) xi in [0, 1], eta in [0, 1] and eta + xi <= 1, i = 1, ..., M    
    to approximate
        int_{Omega_t} f(xi, eta) dxi deta approx sum_{i=1}^{M} w_{i} f(xi_{i}, eta_{i})

    Usage
    -----
        N = 1
        xi, eta, w = gauss_legendre_quadrature_2D_triangle(N) 
    
    Parameters
    ----------
    N : int
        The quadrature degree, it must be a positive number, i.e., N > 0.
        NOTE: For now only N = 1 is implemented.
    
    Returns
    -------
    xi : numpy.array(Float64), size (M, )
        The xi coordinates of the M quadrature nodes associated to the M quadrature 
        weights w_{i}, i = 1, ..., M.
    eta : numpy.array(Float64), size (M, )
        The eta coordinates of the M quadrature nodes associated to the M quadrature 
        weights w_{i}, i = 1, ..., M.
    w : numpy.array(Float64), size (M, )
        The M quadrature weights associated to the M quadrature nodes (xi_{i}, eta_{i}), i = 1, ..., M.
    """
    if N > 1:
        raise Exception("Gauss quadrature on triangles implemented only for N=1")
    
    # We generate the nodes and weights by hand for the case N =1,
    # the only case implemented
    xi = numpy.array([0.0, 1.0, 0.0])
    eta = numpy.array([0.0, 0.0, 1.0])
    w = numpy.array([1.0/6.0, 1.0/6.0, 1.0/6.0])

    return xi, eta, w

def gauss_legendre_quadrature_1D(N):
    """Returns N Gauss-Legendre quadrature weights and nodes for 1D integration.

    Returns N Gauss-Legendre quadrature weights
        w_{i}, i = 1, ..., N
    and N nodes
        xi_{i} in [-1, 1], i = 1, ..., N    
    to approximate
        int_{-1}^{1} f(xi) dxi approx sum_{i=1}^{N} w_{i} f(xi_{i})

    Usage
    -----
        N = 2
        x, w = gauss_legendre_quadrature_1D(N) 
    
    Parameters
    ----------
    N : int
        The number of quadrature nodes, it must be a positive number, i.e., N > 0.
    
    Returns
    -------
    xi : numpy.array(Float64), size (N, )
        The N quadrature nodes associated to the N quadrature weights w_{i}, i = 1, ..., N.
    w : numpy.array(Float64), size (N, )
        The N quadrature weights associated to the N quadrature nodes xi_{i}, i = 1, ..., N.
    """

    # Use the existing function in numpy
    xi, w = numpy.polynomial.legendre.leggauss(N)

    return xi, w


def gauss_legendre_quadrature_2D_quads(N_xi, N_eta):
    """Returns N_xi x N_eta Gauss-Legendre quadrature weights and nodes for 2D integration over quadrilaterals.

    Returns N_xi x N_eta 2D Gauss-Legendre quadrature weights
        w_{i}, i = 1, ..., N_xi x N_eta
    and N_xi x N_eta nodes
        (xi, eta)_{i} in [-1, 1] x [-1, 1], i = 1, ..., N_xi x N_eta    
    to approximate
        int_{-1}^{1} int_{-1}^{1} f(xi, eta) dxi deta approx sum_{i=1}^{N_xi + N_eta} w_{i} f(xi_{i}, eta_{i})

    Note that:
        w_{m + (n - 1)N_xi} = w_xi_{m} w_eta_{n}, m = 1, ..., N_xi and n = 1, ..., N_eta
        (xi, eta)_{m + (n - 1)N_xi} = (xi_{m}, eta_{n}) m = 1, ..., N_xi and n = 1, ..., N_eta

    Usage
    -----
        N_xi = 2
        N_eta = 2
        xi, eta, w = gauss_legendre_quadrature_2D(N_xi, N_eta) 
    
    Parameters
    ----------
    N_xi : int
        The number of quadrature nodes in the xi direction, it must be a positive number, i.e., N > 0.
    N_eta : int
        The number of quadrature nodes in the eta direction, it must be a positive number, i.e., N > 0.
    
    Returns
    -------
    xi : numpy.array(Float64), size (N_xi * N_eta, )
        The xi coordinate of the N = N_xi*N_eta quadrature nodes associated to the N quadrature 
        weights w_{i}, i = 1, ..., N.
    eta : numpy.array(Float64), size (N_xi * N_eta, )
        The eta coordinate of the N = N_xi*N_eta quadrature nodes associated to the N quadrature 
        weights w_{i}, i = 1, ..., N.
    w : numpy.array(Float64), size (N_xi * N_eta, )
        The N=N_xi * N_eta quadrature weights associated to the N quadrature nodes (xi_{i}, eta_{i}), i = 1, ..., N.
    """

    # The Gauss quadrature in 2D is computed via tensor-product of the two 1D quadratures
    
    # Compute the two 1D quadratures
    xi_1D, w_xi = gauss_legendre_quadrature_1D(N_xi)
    eta_1D, w_eta = gauss_legendre_quadrature_1D(N_eta)

    # Compute the 2D quadrature weights as Kronecker product of the 1D quadrature weights
    w = numpy.kron(w_xi, w_eta)

    # Compute the 2D quadrature nodes as tensor product of the 1D nodes
    xi, eta = numpy.meshgrid(xi_1D, eta_1D)

    return xi.flatten(), eta.flatten(), w.flatten()