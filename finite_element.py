# Python code
# Author: Bruno Turcksin
# Date: 2011-04-01 09:49:14.081467

#----------------------------------------------------------------------------#
## Class finite_element                                                     ##
#----------------------------------------------------------------------------#

"""Contain the mass and gradient matrices"""

import numpy as np

class finite_element  :
  """Contain the mass and gradient matrices in 1D and 2D"""

  def __init__(self,param) :
    self.width = param.width
    self.width_cell = np.array([self.width[0]/param.n_div[0],self.width[1]/
      param.n_div[1]])

#----------------------------------------------------------------------------#

  def build_2D_FE(self) :
    """Build the mass and gradient matrices in 2D."""
   
    self.mass_matrix = self.width_cell[0]*self.width_cell[1]/36.*np.array([
     [4.,2.,2.,1.],[2.,4.,1.,2.],[2.,1.,4.,2.],[1.,2.,2.,4.]])
    self.x_grad_matrix = -self.width_cell[1]/12.*np.array([[2.,1.,2.,1.],
      [1.,2.,1.,2.],[-2.,-1.,-2.,-1.],[-1.,-2.,-1.,-2.]])
    self.y_grad_matrix = -self.width_cell[0]/12.*np.array([[2.,2.,1.,1.],
      [-2.,-2.,-1.,-1.],[1.,1.,2.,2.],[-1.,-1.,-2.,-2.]])

#----------------------------------------------------------------------------#

  def build_1D_FE(self) :
    """Build the mass matrix in 1D."""
   
    self.horizontal_edge_mass_matrix = self.width_cell[0]/6.*np.array([[2.,1.],
     [1.,2.]])
    self.vertical_edge_mass_matrix = self.width_cell[1]/6.*np.array([[2.,1.],
     [1.,2.]])
