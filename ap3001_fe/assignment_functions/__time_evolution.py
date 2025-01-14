import numpy
from scipy.sparse.linalg import spsolve as solve

from ..core import assembly

def evolve_in_time(u_0, f, M, N, delta_t, theta, basis, tol=1e-10, t_step_max = 1000):
    """Evolves the temperatura distribution from the initial condition u_0 until it reaches steady state.

    Computes the evolution of the temperature distribution starting with the initial temperature
    distribution u_0. Evolves using a time step delta_t and the theta-method as time integrator.

    The temperature reaches steady state when the maximum temperature change between two time steps is at most
    as large as the tolerance tol (or the maximum number of time steps, t_step_max, is reached).

    To speed up calculations, the mass matrix, M, the siffness matrix, N, and the heat source, f, are all
    given as input.

    The basis functions, basis, is a data structure that contains the information and functionality
    to implement the essential boundary conditions.
        
    Usage
    -----
        See the Jupyter notebook.
    
    Parameters
    ----------
    u_0 : numpy.array(Float64), size (basis.n_basis, )
        The initial temperature distribution.
    f : numpy.array(Float64), size (basis.n_basis, )
        The source vector representing the discrete heat source.
    M: numpy.array(Float64) size (basis.n_basis, basis.n_basis)
        The mass matrix for the basis functions in basis.
    N: numpy.array(Float64) size (basis.n_basis, basis.n_basis)
        The stiffness matrix for the basis functions in basis.
    theta: Float64, size (single value)
        The theta parameter in the theta-time integrator.
            theta = 0.0 --> forward Euler
            theta = 0.5 --> Crank-Nicholson
            theta = 1.0 --> backward Euler
    basis: ap3001_fe.core.HatBasisFunctions, size (single value)
        The basis functions object representing the basis functions to use to solve the problem.
    tol: Float64, size (single value)
        The tolerance used to decide if the temperature reached stationary state. If the temperature 
        changes less than tol between two time steps, the Temperature is assumed to have reached
        steady state (i.e., equilibrium).
    t_step_max: Int, size (single value)
        The maximum number of time steps to perform. If this number of time steps is reached, time
        evolution ends. This is used to avoid running the simulation for too long, for example, if
        a too small time step is chosen.
    
    Returns
    -------
    t_step: Int, size (single value)
        The number of the last time step computed.
    u_t: numpy.array(Float64), size (t_step, basis.n_basis)
        The temperature distribution at each time step. The temperature distribution at time step t_step is 
        u_t[t_step].
    u_diffs: numpy.array(Float64), size (t_step - 1, )
        The maxmimum temperature difference between two time steps.
    t_steps: numpy.array(Float64), size(t_step, )
        The time instants associated to each time step.
    """

    # After semi-discretising your time-dependent PDE in space you obtain a 
    # system of ODEs of the form
    #   M du/dt = -N u + f
    # with u|_Gamma_1 = g_1 and u_|Gamma_3 = g_3

    # Implementing the theta-method to time integrate your system of ODEs
    # in order to go from the solution u^n at time instant t^n to the 
    # the solution u^(n+1) at the next time instant t^(n+1)
    # gives a system of equations with the following form
    #   A u^(n+1) = B u^n + f
    #
    # Note 1: in this case f is constant in time, therefore 
    #   (1+theta)f^(n+1) + f^(n) =(1+theta)f + f = f
    #
    # Note 2: ignore the essential boundary conditions for now, these will be set below,
    #         i.e., consider the system without the essential boundary conditions at this stage.
    A = M/delta_t + theta*N
    B = M/delta_t - N*(1 - theta)

    # Impose boundary conditions in A
    A_bc = assembly.impose_boundary_conditions_matrix(A, basis)

    # Initialise the time step procedure
    u_n = u_0 # initialize the current time step solution as the initial condition
    u_n_1 = u_0  # initialize the next step as the initial condition
    u_diff_max = 1.0  # initialize the difference  between two consecutive time steps
    t_step = 1  # initialize the time step counter

    # Initialize storage of solution at all time steps
    u_t = [u_n]
    t_steps = [0.0]
    u_diffs = []

    # Advance the solution in time until we reach the steady-state, i.e., 
    #   u_diff_max = max( |u_n_1 - u_n| ) < tol << 1
    # or we reached the maximum number of time steps allowed, t_step_max
    # (we do not want to wait forever). 
    while (u_diff_max > tol) and (t_step < t_step_max): 
        # Initialize step
        u_n = u_n_1

        # Compute current time instant
        t = t_step * delta_t

        # Compute B u^n + f^(n)
        b = (B @ u_n) + f

        # Impose boundary conditions on b
        b_bc = assembly.impose_boundary_conditions_vector(b, basis)

        # Solve the system to get the solution at the new time step
        u_n_1 = solve(A_bc, b_bc)
        
        # Compute the difference between the two time steps
        u_diff = u_n_1 - u_n 
        u_diff_max = numpy.abs(u_diff).max()

        # Update time step 
        t_step += 1

        # Save solution at current time step, the time instant, and the error
        u_t.append(u_n_1)
        t_steps.append(t)
        u_diffs.append(u_diff_max)

    return t_step, u_t, u_diffs, t_steps