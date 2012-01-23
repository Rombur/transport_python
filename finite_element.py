# Python code
# Author: Bruno Turcksin
# Date: 2011-04-01 09:49:14.081467

#----------------------------------------------------------------------------#
## Class finite_element                                                     ##
#----------------------------------------------------------------------------#

"""Contain the mass and gradient matrices"""

import numpy as np

class finite_element(object) :
  """Contain the mass and gradient matrices in 1D and 2D : 
    1--3
    |  |
    0--2"""

  def __init__(self,param) :

    super(finite_element,self).__init__()
    self.width = param.width
    self.width_cell = np.array([self.width[0]/param.n_div[0],self.width[1]/
      param.n_div[1]])
    self.n_dofs_per_cell = 4

#----------------------------------------------------------------------------#

  def build_2D_FE(self) :
    """Build the mass and gradient matrices in 2D."""
   
    self.mass_matrix = self.width_cell[0]*self.width_cell[1]/36.*np.array([
     [4.,2.,2.,1.],[2.,4.,1.,2.],[2.,1.,4.,2.],[1.,2.,2.,4.]])
    self.x_grad_matrix = -self.width_cell[1]/12.*np.array([[2.,1.,2.,1.],
      [1.,2.,1.,2.],[-2.,-1.,-2.,-1.],[-1.,-2.,-1.,-2.]])
    self.y_grad_matrix = -self.width_cell[0]/12.*np.array([[2.,2.,1.,1.],
      [-2.,-2.,-1.,-1.],[1.,1.,2.,2.],[-1.,-1.,-2.,-2.]])
    self.stiffness_matrix = self.width_cell[1]/(6*self.width_cell[0])*\
        np.array([[2.,1.,-2.,-1.],[1.,2.,-1.,-2.],[-2.,-1.,2.,1.],\
        [-1.,-2.,1.,2.]]) + self.width_cell[0]/(6*self.width_cell[1])*\
        np.array([[2.,-2.,1.,-1.],[-2.,2.,-1.,1.],[1.,-1.,2.,-2.],\
        [-1.,1,-2.,2.]])            

#----------------------------------------------------------------------------#

  def build_1D_FE(self) :
    """Build the mass matrix in 1D."""
   
    self.horizontal_edge_mass_matrix = self.width_cell[0]/6.*np.array([[2.,1.],
     [1.,2.]])
    self.vertical_edge_mass_matrix = self.width_cell[1]/6.*np.array([[2.,1.],
     [1.,2.]])
    
    self.compute_edge_deln_matrix()
    self.compute_across_edge_deln_matrix()

#----------------------------------------------------------------------------#

  def compute_edge_deln_matrix(self) :
    """Compute the matrices (b,n nabla b) for the edges."""

    ratio = self.width_cell[1]/(6*self.width_cell[0])
    left = ratio*np.array([[2.,1.,0.,0.],[1.,2.,0.,0.],[-2.,-1.,0.,0],
      [-1.,-2.,0.,0.]])

    right = ratio*np.array([[0.,0.,-2.,-1.],[0.,0.,-1.,-2.],[0.,0.,2.,1.],
      [0.,0.,1.,2.]])

    ratio = self.width_cell[0]/(6*self.width_cell[1])
    bottom = ratio*np.array([[2.,0.,1.,0.],[-2.,0.,-1.,0.],[1.,0.,2.,0],
      [-1.,0.,-2.,0.]])

    top = ratio*np.array([[0.,-2.,0.,-1.],[0.,2.,0.,1.],[0.,-1.,0.,-2.],
      [0.,1.,0.,2.]])

    self.edge_deln_matrix = {'left' : left, 'right' : right, 'bottom' :
        bottom, 'top' : top}

#----------------------------------------------------------------------------#

  def compute_across_edge_deln_matrix(self) :
    """Compute the matricex (b^+, n nabla b^-) for the edges."""

    ratio = self.width_cell[1]/(6*self.width_cell[0])
    left = ratio*np.array([[0.,0.,2.,1.],[0.,0.,1.,2.],[0.,0.,-2.,-1],
      [0.,0.,-1.,-2.]])

    right = ratio*np.array([[-2.,-1.,0.,0.],[-1.,-2.,0.,0.],[2.,1.,0.,0.],
      [1.,2.,0.,0.]])

    ratio = self.width_cell[0]/(6*self.width_cell[1])
    bottom = ratio*np.array([[0.,2.,0.,1.],[0.,-2.,0.,-1.],[0.,1.,0.,2.],
      [0.,-1.,0.,-2.]])

    top = ratio*np.array([[-2.,0.,-1.,0.],[2.,0.,1.,0.],[-1.,0.,-2.,0.],
      [1.,0.,2.,0.]])

    self.across_edge_deln_matrix = {'left' : left, 'right' : right, 'bottom' :
        bottom, 'top' : top}
