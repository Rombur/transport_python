# Python code
# Author: Bruno Turcksin
# Date: 2012-06-22 10:05:10.997713

#----------------------------------------------------------------------------#
## Class DOF_HANDLER                                                        ##
#----------------------------------------------------------------------------#

"""Build the cells, the spatial discretization and the sweep ordering."""

import numpy as np
import CELL
import FINITE_ELEMENT
import SPECTRAL_VOLUME

class DOF_HANDLER(object) :
  """Build the cells, the spatial discretization associated to each cell and
  the sweep ordering."""

  def __init__(self,param) :

    super(DOF_HANDLER,self).__init__()

    self.bottom = 0.
    self.right = 0.
    self.top = 0.
    self.left = 0.
    self.mesh = []  
    cell_id = 0
    first_dof = 0
    for j in range(param.n_y) :
      for i in range(param.n_x) :      
        width = np.array([param.x_width[i],param.y_width[j]])
        if param.discretization=='FE' :
          sd = FINITE_ELEMENT.FINITE_ELEMENT(width)
        else :
          sd = SPECTRAL_VOLUME.SPECTRAL_VOLUME(width)
        mat_id = param.mat_id[i,j]
        src_id = param.src_id[i,j]
        self.mesh.append(CELL.CELL(cell_id,width,np.array([self.right,self.top]),
          param.sig_t[mat_id],param.sig_s[mat_id],param.src[src_id],sd,first_dof,
          first_dof+sd.n_dof_per_cell))
        cell_id += 1
        first_dof += sd.n_dof_per_cell
        if j==0 :
          self.right += width[0]
      self.top += width[1]   
    
    self.n_dof = first_dof

#----------------------------------------------------------------------------#

  def Compute_sweep_ordering(self,quad,param) :
    """Compute the sweep ordering for each direction in the quadrature."""

    self.sweep_ordering = [[] for i in range(quad.n_dir)]
    for idir in range(quad.n_dir) :
      omega_x = quad.omega[idir,0]
      omega_y = quad.omega[idir,1]
      idir_sweep_ordering = []
      if omega_x>0.:
        if omega_y>0. :
          for i in range(param.n_cells) :
            idir_sweep_ordering.append(i)
        else :
          for j in range(param.n_y-1,-1,-1) :
            for i in range(param.n_x) :
              idir_sweep_ordering.append(j*param.n_x+i)
      else :
        if omega_y>0. :
          for j in range(param.n_y) :
            for i in range(param.n_x-1,-1,-1) :
              idir_sweep_ordering.append(j*param.n_x+i)
        else :
          for j in range(param.n_y-1,-1,-1) :
            for i in range(param.n_x-1,-1,-1) :
              idir_sweep_ordering.append(j*param.n_x+i)
