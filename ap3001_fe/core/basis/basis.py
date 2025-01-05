
import numpy

from ...assignment_functions import *
from ..quadrature.quadrature import *


class HatBasisFunctions:
    def __init__(self, mesh, bcs):
        # Mesh information
        self.mesh = mesh  # locally save the mesh

        # Set the number of basis in the mesh
        self.n_basis = self.__number_of_basis()

        # Get the indices of basis and values at each essential boundary
        self.essential_bcs = self.__get_essential_boundary_conditions(bcs)

        # Set the number of canonical basis
        self.n_canonical_basis = 3  # for piecewise linear hat functions on a triangle, there are are 
                                    # three canonical basis: one associated to each of the three vertices
                                    # of the canonical triangle

        # Compute the full extraction coefficient tensor E
        # E[l, i, j] is the coefficient of local basis j, associated to global basis i, on element l.
        #   phi_j (x, y) = sum_l sum_j E[l, i, j] B_l_j(x, y)
        # or using the canonical basis
        #   phi_j (x, y) = sum_l sum_j E[l, i, j] B_c_j( Phi_l^-1 (x, y) )
        self.E_full = self.__compute_full_extraction_coefficient_tensor_hat_function_triangle()

        # Compute the extraction coefficient tensor with respect to the active basis on each element 
        # Now E_active_basis[k, i, j] is the coefficient of local basis j, associated to active global basis i, on element l.
        # active global basis i is the global basis with index element_active_basis[k, i], note that i = 0, ..., (n_active_basis_in_element - 1)
        # In this way it becomes inneficient to compute each global basis separately as we were doing before
        #   phi_j (x, y) = sum_l sum_j E[l, i, j] B_l_j(x, y)
        # or using the canonical basis
        #   phi_j (x, y) = sum_l sum_j E[l, i, j] B_c_j( Phi_l^-1 (x, y) )
        # We can now only efficiently compute the contribution of each local element basis to all global basis.
        # This is not an issue, because this is precisely what we need.
        self.element_active_basis_idx, self.E_active_basis = self.__compute_efficient_extraction_coefficient_tensor_hat_function_triangle()


    def init_solution(self):
        return numpy.zeros(self.n_basis)

    def __number_of_basis(self):
        n_vertices = self.mesh.number_of_vertices()  # number of vertices on the mesh
        n_basis = n_vertices  # for piecewise linear hat functions on a triangular mesh, there are
        return n_basis 
    
                        # as many basis as vertices in the mesh
    def basis_canonical(self, xi, eta):
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
            mesh = ap3001_lib.Mesh('square', refinement=1) 
            xi = numpy.array([0.0, 1.0, 0.0])
            eta = numpy.array([0.0, 0.0, 1.0])
            hat_basis = ap3001_lib.HatBasisFunctions(mesh)
            B_c_eval = hat_basis.basis_canonical(xi, eta)
        
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
        return basis_canonical_triangle(xi, eta)
    

    def grad_basis_canonical(self, xi, eta):
        """Evaluates the gradient of the canonical basis that enable the construction of hat functions.

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
            mesh = ap3001_lib.Mesh('square', refinement=1) 
            xi = numpy.array([0.0, 1.0, 0.0])
            eta = numpy.array([0.0, 0.0, 1.0])
            hat_basis = ap3001_lib.HatBasisFunctions(mesh)
            dB_c_eval_dxi, d_B_c_eval_deta = hat_basis.grad_basis_canonical(xi, eta)
        
        Parameters
        ----------
        xi : numpy.array(Float64), size (M, )
            The xi coordinates of the M points where to evaluate the three canonical basis.
        eta : numpy.array(Float64), size (M, )
            The eta coordinates of the M points where to evaluate the three canonical basis.
        
        Returns
        -------
        dB_c_eval_dxi : numpy.array(Float64), size (3, M)
            The array containing the evaluation of the derivative with respect to xi of the
            three canonical basis on the M nodes (xi_i, eta_i).
        dB_c_eval_deta : numpy.array(Float64), size (3, M)
            The array containing the evaluation of the derivative with respect to eta of the
            three canonical basis on the M nodes (xi_i, eta_i).
        """
        return grad_basis_canonical_triangle(xi, eta)


    def quadrature(self):
        """Returns the Gauss quadrature weights and nodes suitable for this basis.

        Returns the three 2D Gauss quadrature weights 
            w_{i}, i = 1, 2, 3
        associated to the nodes    
            (xi, eta)_{i} in the canonical triangle (Omega_t) xi in [0, 1], eta in [0, 1] and eta + xi <= 1, i = 1, 2, 3    
        to approximate
            int_{Omega_t} f(xi, eta) dxi deta approx sum_{i=1}^{N} w_{i} f(xi_{i}, eta_{i})

        Usage
        -----
            mesh = ap3001_lib.Mesh('square', refinement=1) 
            xi = numpy.array([0.0, 1.0, 0.0])
            eta = numpy.array([0.0, 0.0, 1.0])
            hat_basis = ap3001_lib.HatBasisFunctions(mesh)
            xi, eta, w = hat_basis.quadrature() 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        xi : numpy.array(Float64), size (3, )
            The xi coordinates of the three quadrature nodes associated to the quadrature 
            weights w_{i}, i = 1, 2, 3.
        eta : numpy.array(Float64), size (3, )
            The eta coordinates of the three quadrature nodes associated to the quadrature 
            weights w_{i}, i = 1, 2, 3.
        w : numpy.array(Float64), size (3, )
            The three quadrature weights associated to the three quadrature nodes (xi_{i}, eta_{i}), i = 1, 2, 3.
        """
        return gauss_quadrature_2D_triangle(1)


    def __compute_full_extraction_coefficient_tensor_hat_function_triangle(self):  
        # Extract mesh information 
        n_elements = self.mesh.number_of_elements()  

        # Pre-allocate memory for the extraction coefficient tensor
        E_full = numpy.zeros([n_elements, self.n_basis, self.n_canonical_basis])

        # Fill the extraction coefficient tensor with non-zero coefficients
        for element_idx, global_bases_idxs in enumerate(self.mesh.elements):
            # Since there is one global basis per vertex of the mesh, we number the 
            # basis in the same way as the vertices, therefore basis i is associated 
            # to vertex i.
            # We do the same for the local and canonical basis numbering, therefore 
            # vertex 1 of element is associated to the local and canonical bases 1
            # of the element. This makes life easier.
            for local_basis_idx, global_basis_idx in enumerate(global_bases_idxs):
                E_full[element_idx, global_basis_idx, local_basis_idx] = 1.0

        return E_full


    def __compute_efficient_extraction_coefficient_tensor_hat_function_triangle(self):
        # Extract mesh information 
        n_elements = self.mesh.number_of_elements() 
        elements = self.mesh.elements

        # Compute the indices of the basis that are nonzero in each element 
        # i.e., the indices of the active basis in each element
        element_active_basis_idx = self.mesh.elements

        # Compute the extraction coefficient matrix only for the active basis on each element
        n_active_basis_per_elements = 3  # for this case, we will always have 3 active basis per element
                                         # in other cases it is more complicated, for example, each element 
                                         # may have a different number of active basis
        
        # Construct the extraction coefficient tensor with respect to the active basis on each element 
        # Now E_active_basis[k, i, j] is the coefficient of local basis j, associated to active global basis i, on element l.
        # active global basis i is the global basis with index element_active_basis[k, i], note that i = 0, ..., (n_active_basis_in_element - 1)
        # In this way it becomes inneficient to compute each global basis separately as we were doing before
        #   phi_j (x, y) = sum_l sum_j E[l, i, j] B_l_j(x, y)
        # or using the canonical basis
        #   phi_j (x, y) = sum_l sum_j E[l, i, j] B_c_j( Phi_l^-1 (x, y) )
        # We can now only efficiently compute the contribution of each local element basis to all global basis.
        # This is not an issue, because this is precisely what we need.
        E_active_basis = numpy.zeros([n_elements, n_active_basis_per_elements, self.n_canonical_basis])
        
        for element_idx in numpy.arange(0, len(elements)):
            E_active_basis[element_idx, 0, 0] = 1.0
            E_active_basis[element_idx, 1, 1] = 1.0
            E_active_basis[element_idx, 2, 2] = 1.0

        return element_active_basis_idx, E_active_basis
    

    def __get_essential_boundary_conditions(self, bcs):
        # Initialize the dictionary with essential boundary conditions
        # The labels are the ones used in the input dictionary gammas 
        essential_bc_conditions = dict.fromkeys(list(bcs))

        # First get all the indices of the boundary vertices
        boundary_vertices_idx = self.mesh.boundary_vertices  # indices of all vertices on the boundary

        # Get x and y coordinates of all the vertices on the boundary 
        x_boundary_vertices = self.mesh.vertices[boundary_vertices_idx, 0]
        y_boundary_vertices = self.mesh.vertices[boundary_vertices_idx, 1]

        # Loop over each of the boundary conditions and
        #   - find the basis (vertices) in that boundary
        #   - compute the boundary value

        for boundary_key in bcs:
            # Initialize dictionary with boundary condition
            essential_bc_conditions[boundary_key] = {'indices': None, 'values': None}

            # Extract boundary condition information
            on_boundary_function = bcs[boundary_key]['on_boundary_function']  # function that evaluates to True if point (x, y) lies on the boundary or False if not
            boundary_g = bcs[boundary_key]['g']  # function that returns the value of the solution at the boundary

            # Find the indices of the vertices on the current boundary
            on_boundary_mask = on_boundary_function(x_boundary_vertices, y_boundary_vertices)
            vertices_on_current_boundary_idx = boundary_vertices_idx[on_boundary_mask]
            essential_bc_conditions[boundary_key]['indices'] = vertices_on_current_boundary_idx

            # Compute the value of the solution on those vertices
            x_on_current_boundary = self.mesh.vertices[vertices_on_current_boundary_idx, 0]
            y_on_current_boundary = self.mesh.vertices[vertices_on_current_boundary_idx, 1]
            essential_bc_conditions[boundary_key]['values'] = boundary_g(x_on_current_boundary, y_on_current_boundary)

        return essential_bc_conditions