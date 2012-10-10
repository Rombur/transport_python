# Python code
# Author: Bruno Turcksin
# Date: 2012-06-22 09:56:34.157051

#----------------------------------------------------------------------------#
## Class CELL                                                               ##
#----------------------------------------------------------------------------#

"""Contain all the data needed to define a cell."""

import numpy as np
                                                                
class CELL(object) :
  """Class which contains all the data needed to define a class."""

  def __init__(self,cell_id,width,v_0,sig_t,sig_s,src,discretization,first_dof,
      last_dof) :

    super(CELL,self).__init__()
    
    self.cell_id = cell_id
    self.width = width
    self.sigma_t = sig_t
    self.sigma_s = sig_s
    self.src = src
    self.sd = discretization
    self.first_dof = first_dof
    self.last_dof = last_dof
    self.v0 = np.array([v_0[0],v_0[1]])
    self.v1 = np.array([v_0[0]+width[0],v_0[1]])
    self.v2 = np.array([v_0[0]+width[0],v_0[1]+width[1]])
    self.v3 = np.array([v_0[0],v_0[1]+width[1]])
