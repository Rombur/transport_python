# Python code
# Author: Bruno Turcksin
# Date: 2011-04-01 09:49:14.081467

#----------------------------------------------------------------------------#
## Class FINITE_ELEMENT                                                     ##
#----------------------------------------------------------------------------#

"""Contain the mass and gradient matrices"""

import numpy as np

class FINITE_ELEMENT(object) :
  """Contain the mass and gradient matrices in 1D and 2D"""

  def __init__(self,width) :

    super(FINITE_ELEMENT,self).__init__()

    self.width = width
    self.n_dof_per_cell = 4

#----------------------------------------------------------------------------#

  def Build_2d_fe(self) :
    """Build the mass and gradient matrices in 2D."""

    self.mass_matrix = self.width[0]*self.width[1]/36.*np.array([
      [4.,2.,1.,2.],[2.,4.,2.,1.],[1.,2.,4.,2.],[2.,1.,2.,4.]])
    self.x_grad_matrix = self.width[1]/12.*np.array([[-2.,-2.,-1.,-1.],
      [2.,2.,1.,1.],[1.,1.,2.,2.],[-1.,-1.,-2.,-2.]])
    self.y_grad_matrix = self.width[0]/12.*np.array([[-2.,-1.,-1.,-2.],
      [-1.,-2.,-2.,-1.],[1.,2.,2.,1.],[2.,1.,1.,2.]])

#----------------------------------------------------------------------------#

  def Build_1d_fe(self) :
    """Build the mass matrix in 1D."""
   
# Edge numbering:
#   - bottom = 0
#   - right = 1
#   - top = 2
#   - left = 3

    self.downwind = [np.zeros((2,2)) for i in xrange(4)]
    self.upwind = [np.zeros((2,2)) for i in xrange(4)]
    h_ratio = self.width[0]/6.
    v_ratio = self.width[1]/6.

# Downwind bottom edge
    self.downwind[0][0,0] = 2.*h_ratio
    self.downwind[0][0,1] = 1.*h_ratio
    self.downwind[0][1,0] = 1.*h_ratio
    self.downwind[0][1,1] = 2.*h_ratio
# Downwind right edge
    self.downwind[1][1,1] = 2.*v_ratio
    self.downwind[1][1,2] = 1.*v_ratio
    self.downwind[1][2,1] = 1.*v_ratio
    self.downwind[1][2,2] = 2.*v_ratio
# Downwind top edge
    self.downwind[2][2,2] = 2.*h_ratio
    self.downwind[2][2,3] = 1.*h_ratio
    self.downwind[2][3,2] = 1.*h_ratio
    self.downwind[2][3,3] = 2.*h_ratio
# Downwind left edge
    self.downwind[3][0,0] = 2.*v_ratio
    self.downwind[3][0,3] = 1.*v_ratio
    self.downwind[3][3,0] = 1.*v_ratio
    self.downwind[3][3,3] = 2.*v_ratio

# Upwind bottom edge
    self.upwind[0][0,2] = 1.*h_ratio
    self.upwind[0][0,3] = 2.*h_ratio
    self.upwind[0][1,2] = 2.*h_ratio
    self.upwind[0][1,3] = 1.*h_ratio
# Upwind right edge
    self.upwind[1][1,0] = 2.*v_ratio
    self.upwind[1][1,3] = 1.*v_ratio
    self.upwind[1][2,0] = 1.*v_ratio
    self.upwind[1][2,3] = 2.*v_ratio
# Upwind top edge
    self.upwind[2][2,0] = 1.*h_ratio
    self.upwind[2][2,1] = 2.*h_ratio
    self.upwind[2][3,0] = 2.*h_ratio
    self.upwind[2][3,1] = 1.*h_ratio
# Upwind right edge
    self.upwind[3][0,1] = 2.*v_ratio
    self.upwind[3][0,2] = 1.*v_ratio
    self.upwind[3][3,1] = 1.*v_ratio
    self.upwind[3][3,2] = 2.*v_ratio
