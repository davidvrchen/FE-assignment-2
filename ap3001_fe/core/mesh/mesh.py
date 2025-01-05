import numpy
import scipy

import skfem
from skfem.visuals.matplotlib import plot as skplot
from skfem.visuals.matplotlib import plot3 as skplot3, show

from ...assignment_functions import *

class Mesh:
    def __init__(self, type_of_mesh: str, refinement=0):
        # Generate the mesh
        if type_of_mesh == "circle":
            self._mesh = skfem.MeshTri1().init_circle().refined(refinement)

        elif type_of_mesh == "square":
            self._mesh = skfem.MeshTri1().refined(refinement)

        else:
            print("Error: domain type not recognized")
            
        # Make quick access to mesh entities
        self.vertices = self._mesh.p.T  # the coordinates of the vertices: self.vertices[k, :] = [x_k, y_k]
        self.edges = self._mesh.facets.T  # the indices of the vertices that make up each edge: self.edges[k, :] = [V_k_start, V_k_end], where V_k_start and V_k_end are indices of the vertices at the start and end of edge k
        self.boundary_edges = self.edges[self._mesh.boundary_facets()]  # the indices of the vertices that make up each edge on the boundary
        self.boundary_vertices = numpy.unique(self.boundary_edges.flatten())  # the indices of the vertices on the boundary
        self.elements = self._mesh.t.T  # the indices of the vertices that make up each element (triangle) in clockwise order: self.elements[k, :] = [V_k_1, V_k_2, V_k_3]
        self.type_of_mesh = type_of_mesh
    

    def number_of_vertices(self):
        """ The number of vertices in the mesh.

         Usage
        -----
            mesh = ap3001_lib.Mesh('square',refinement=1)
            xi = numpy.array([0.0, 1.0, 0.0])
            eta = numpy.array([0.0, 0.0, 1.0])
            n_vertices = mesh.number_of_vertices()

        Parameters
        ----------
        None

        Returns
        -------
        n_vertices : int, size (single value)
            The number of vertices in the mesh.
        """
        return len(self.vertices)
    

    def number_of_elements(self):
        """ The number of elements in the mesh.

         Usage
        -----
            mesh = ap3001_lib.Mesh('square',refinement=1)
            xi = numpy.array([0.0, 1.0, 0.0])
            eta = numpy.array([0.0, 0.0, 1.0])
            n_elements = mesh.number_of_elements()

        Parameters
        ----------
        None

        Returns
        -------
        n_elements : int, size (single value)
            The number of elements in the mesh.
        """
        return len(self.elements)
    

    def mapping(self, xi, eta, element_idx: int):
        """Evaluates mapping between the canonical triangle element and the physical 
        triangle element_idx.

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
            mesh = ap3001_lib.Mesh('square',refinement=1)
            xi = numpy.array([0.0, 1.0, 0.0])
            eta = numpy.array([0.0, 0.0, 1.0])
            element_idx = 4
            x, y = mesh.mapping(xi, eta, element_idx)

            matplotlib.pyplot.plot(x, y)
            matplotlib.pyplot.show()
        
        Parameters
        ----------
        xi : numpy.array(Float64), size (M, )
            The xi coordinates of the M points where to evaluate the mapping Phi.
        eta : numpy.array(Float64), size (M, )
            The eta coordinates of the M points where to evaluate the mapping Phi.
        element_idx: int size (single value)
            The index of the element (triangle) in the mesh to use.
        
        Returns
        -------
        x : numpy.array(Float64), size (M,)
            The vector containing  x physical coordinates of the points inside the triangle
            that are the image of the canonical points (xi_i, eta_i) via the mapping Phi.
        y : numpy.array(Float64), size (M,)
            The vector containing the y physical coordinates of the points inside the triangle
            that are the image of the canonical points (xi_i, eta_i) via the mapping Phi.
        """
        vertices_idx = self.elements[element_idx]  # indices of the vertices that make up the element
        vertices_coordinates = self.vertices[vertices_idx]  # the (x, y) coordinates of the vertices of the element

        return mapping_triangle(xi, eta, vertices_coordinates)
    

    def jacobian(self, xi, eta, element_idx: int):
        """Evaluates Jacobian matrix of the mapping between the canonical 
        triangle element and the physical triangle with index element_idx.

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
            mesh = ap3001_lib.Mesh('square',refinement=1)
            xi = numpy.array([0.0, 1.0, 0.0])
            eta = numpy.array([0.0, 0.0, 1.0])
            element_idx = 4
            J = mesh.jacobian(xi, eta, element_idx)
        
        Parameters
        ----------
        xi : numpy.array(Float64), size (M, )
            The xi coordinates of the M points where to evaluate the mapping Phi.
        eta : numpy.array(Float64), size (M, )
            The eta coordinates of the M points where to evaluate the mapping Phi.
        element_idx: int size (single value)
            The index of the element (triangle) in the mesh to use.
        
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
        vertices_idx = self.elements[element_idx]  # indices of the vertices that make up the element
        vertices_coordinates = self.vertices[vertices_idx]  # the (x, y) coordinates of the vertices of the element

        return jacobian_triangle(xi, eta, vertices_coordinates)
    

    def jacobian_transpose_times_jacobian(self, xi, eta, element_idx):
        """Evaluates the metric matrix G = (J^T) J of the mapping between 
        the canonical triangle element and the physical triangle with index element_idx.

        Computes the metric matrix G = (J^T) J, where J is the Jacobian matrix
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
            
        Returns the metric matrix G evaluated at the points (xi_i, eta_i)
            G[i, :, :] = J[i, :, :] (J[i, :, :]^T) 
        
        where J[i, :, :] is the Jacobian matrix evaluated at point (xi_i, eta_i).
            
        Usage
        -----
            mesh = ap3001_lib.Mesh('square', refinement=1)
            xi = numpy.array([0.0, 1.0, 0.0])
            eta = numpy.array([0.0, 0.0, 1.0])
            element_idx = 4
            G = mesh.jacobian_transpose_times_jacobian(xi, eta, element_idx)
            
        Parameters
        ----------
        xi : numpy.array(Float64), size (M, )
            The xi coordinates of the M points where to evaluate the mapping Phi.
        eta : numpy.array(Float64), size (M, )
            The eta coordinates of the M points where to evaluate the mapping Phi.
        element_idx: int size (single value)
            The index of the element (triangle) in the mesh to use.
        
        Returns
        -------
        G : numpy.array(Float64), size (M, 2, 2)
            The metric matrix G = (J^T) J evaluated at the points (xi_i, eta_i)
                G[i, :, :] = (J[i, :, :]^T) J[i, :, :]
            where J[i, :, :] is the Jacobian matrix evaluated at point (xi_i, eta_i).
        """

        # Check if inputs have the correct shape
        if (len(xi.shape) > 1) or (len(eta.shape) > 1):
            raise Exception("The input coordinates must be vectors, not arrays.")
        
        if (len(xi) != len(eta)):
            raise Exception("The input coordinates must have the same number of coordinates.")

        # Compute the G matrix G = (J^T) J
        J = self.jacobian(xi, eta, element_idx)
        G = numpy.linalg.matrix_transpose(J) @ J  # @ is just the operator in numpy for matrix multiplication
        
        return G


    def inv_jacobian_transpose_times_jacobian(self, xi, eta, element_idx):
        """Evaluates the inverse of the metric matrix G = (J^T) J of the mapping between 
        the canonical triangle element and the physical triangle with index element_idx.

        Computes the invers of the metric matrix G = (J^T) J, where J is the Jacobian matrix
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
            
        Returns the inverse of the metric matrix G evaluated at the points (xi_i, eta_i)
            inv_G[i, :, :] = ((J[i, :, :]^T) J[i, :, :])^-1
        
        where J[i, :, :] is the Jacobian matrix evaluated at point (xi_i, eta_i).
            
        Usage
        -----
            mesh = ap3001_lib.Mesh('square', refinement=1)
            xi = numpy.array([0.0, 1.0, 0.0])
            eta = numpy.array([0.0, 0.0, 1.0])
            element_idx = 4
            inv_G = mesh.inv_jacobian_transpose_times_jacobian(xi, eta, element_idx)
        
        Parameters
        ----------
        xi : numpy.array(Float64), size (M, )
            The xi coordinates of the M points where to evaluate the mapping Phi.
        eta : numpy.array(Float64), size (M, )
            The eta coordinates of the M points where to evaluate the mapping Phi.
        element_idx: int size (single value)
            The index of the element (triangle) in the mesh to use. 
        
        Returns
        -------
        inv_G : numpy.array(Float64), size (M, 2, 2)
                The inverse of the metric matrix G = (J^T) J, i.e., G^-1, evaluated at the points (xi_i, eta_i)
                    inv_G[i, :, :] = (G[i, :, :])^-1
                where G[i, :, :] is metric matrix evaluated at point (xi_i, eta_i), and J is the
                Jacobian matrix.
        """

        # Check if inputs have the correct shape
        if (len(xi.shape) > 1) or (len(eta.shape) > 1):
            raise Exception("The input coordinates must be vectors, not arrays.")
        
        if (len(xi) != len(eta)):
            raise Exception("The input coordinates must have the same number of coordinates.")

        # Evaluate the metric matrix G
        G = self.jacobian_transpose_times_jacobian(xi, eta, element_idx)

        # Compute the inverse
        inv_G = numpy.linalg.inv(G)

        return inv_G


    def sqrt_det_jacobian_transpose_times_jacobian(self, xi, eta, element_idx):
        """Evaluates the square root of the absolute value of the determinant of the 
        metric matrix G = (J^T) J of the mapping between the canonical triangle element 
        and the physical triangle with index element_idx.

        Computes the square root of the determinant of the inverse of the metric matrix
            sqrt(abs(det(G)))

        with 
            G = (J^T) J
        
        and J the Jacobian
              -                       -
        J =  | dPhi^x/dxi  dPhi^x/deta |
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
            
        Returns the square root of the absolute value of the determinant of the 
        metric matrix G = (J^T) J at the points (xi_i, eta_i)
            det_G[i] = sqrt(abs(det(G[i, :, :])))

        NOTE: if the physical triangle is in R^2 then det_G[i] = det_J[i],
            as generated by det_jacobian_triangle(xi, eta, vertices). Otherwise,
            we are unable to compute the determinant of the Jacobian, because it
            will not be a square matrix. We can always compute sqrt(abs(det(G))),
            which makes it more general and more useful. For your use now, you can see
            both as the same.
            
        Usage
        -----
            mesh = ap3001_lib.Mesh('square', refinement=1)
            xi = numpy.array([0.0, 1.0, 0.0])
            eta = numpy.array([0.0, 0.0, 1.0])
            element_idx = 4
            det_G = mesh.sqrt_det_jacobian_transpose_times_jacobian(xi, eta, element_idx)
        
        Parameters
        ----------
        xi : numpy.array(Float64), size (M, )
            The xi coordinates of the M points where to evaluate the mapping Phi.
        eta : numpy.array(Float64), size (M, )
            The eta coordinates of the M points where to evaluate the mapping Phi.
        element_idx: int size (single value)
            The index of the element (triangle) in the mesh to use.
        
        Returns
        -------
        det_G : numpy.array(Float64), size (M,)
                The square root of the absolute value of the determinant of the 
                metric matrix G = (J^T) J, evaluated at the points (xi_i, eta_i)
                det_G[i] = sqrt(abs(det(G[i, :, :]))).
        """

        # Check if inputs have the correct shape
        if (len(xi.shape) > 1) or (len(eta.shape) > 1):
            raise Exception("The input coordinates must be vectors, not arrays.")
        
        if (len(xi) != len(eta)):
            raise Exception("The input coordinates must have the same number of coordinates.")
        
        # Compute the inverse of the metrix matrix
        G = self.jacobian_transpose_times_jacobian(xi, eta, element_idx)

        # Compute its determinant
        det_G = numpy.linalg.det(G)

        # Compute the absolute value and then its square root
        sqrt_abs_det_G = numpy.sqrt(numpy.abs(det_G))

        return sqrt_abs_det_G


    def det_jacobian(self, xi, eta, element_idx):
        """Evaluates the absolute value of the determinant of the Jacobian matrix of the mapping 
        between the canonical triangle element and the physical triangle with index element_idx.

        Computes the absolute value of determinant of the Jacobian matrix
                          -                       -
        det J = abs( det | dPhi^x/dxi  dPhi^x/deta | )
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
            
        Returns the determinant of the Jacobian matrix evaluated at the points (xi_i, eta_i)
            det(J[i, :, :])
            
        Usage
        -----
            mesh = ap3001_lib.Mesh('square', refinement=1)
            xi = numpy.array([0.0, 1.0, 0.0])
            eta = numpy.array([0.0, 0.0, 1.0])
            element_idx = 4
            det_J = mesh.det_jacobian_triangle(xi, eta, element_idx)
        
        Parameters
        ----------
        xi : numpy.array(Float64), size (M, )
            The xi coordinates of the M points where to evaluate the mapping Phi.
        eta : numpy.array(Float64), size (M, )
            The eta coordinates of the M points where to evaluate the mapping Phi.
        element_idx: int size (single value)
            The index of the element (triangle) in the mesh to use.
        
        Returns
        -------
        det_J : numpy.array(Float64), size (M,)
                The absolute value of the determinant of the Jacobian matrix evaluated at the points (xi_i, eta_i)
                det_J[i] = abs(det(J(xi_i, eta_i)))
        """

        # Check if inputs have the correct shape
        if (len(xi.shape) > 1) or (len(eta.shape) > 1):
            raise Exception("The input coordinates must be vectors, not arrays.")
        
        if (len(xi) != len(eta)):
            raise Exception("The input coordinates must have the same number of coordinates.")

        # Evaluate the Jacobian at each of the points
        J = self.jacobian(xi, eta, element_idx)

        # Compute the determinant for each point
        det_J = numpy.abs(numpy.linalg.det(J))

        return det_J
    

    def refine(self, value: int):
        return Mesh(self.type_of_mesh, self.mesh.refined(value))
        
    def draw(self):
        ax = self._mesh.draw()
        ax.set_axis_on()
        return ax

def plot(mesh, u, **kwargs):
    return skplot(mesh._mesh, u, **kwargs)

def plot3d(mesh,u):
    return skplot3(mesh._mesh, u)