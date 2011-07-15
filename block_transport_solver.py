# Python code
# Author: Bruno Turcksin
# Date: 2011-07-10 18:57:42.034268

#----------------------------------------------------------------------------#
## Class block_transport_solver                                             ##
#----------------------------------------------------------------------------#

"""Contain the block solver of the transport equation."""

import numpy as np
import scipy.linalg
import transport_solver as t_s

class block_transport_solver(object) :
  """Solve the transport equation in 2D for cartesian geometry, using a
  block algorithm. Works only with SI."""

  def __init__(self,param,fe,quad,flux_moments,all_psi,x_down,x_up,y_down,\
      y_up,most_normal) :

    super(block_transport_solver,self).__init__()
    self.param = param
    self.fe = fe
    self.quad = quad
    self.flux_moments = flux_moments
    self.all_psi = all_psi
    self.x_down = x_down
    self.x_up = x_up
    self.y_down = y_down
    self.y_up = y_up
    self.most_n = most_normal

#----------------------------------------------------------------------------#

  def solve(self) :
    """Driver for the block sweep. The function is purely virtual."""

    raise NotImplementedError("solve is purely virtual and must be overridden.")

#----------------------------------------------------------------------------#

  def compute_scattering_source(self,x) :
    """Compute the scattering given a flux."""

    self.scattering_src = np.zeros((4*self.param.n_mom*self.param.n_cells))
    for cell in xrange(0,int(self.param.n_cells)) :
# Get i,j pair from a cell
      [i,j] = t_s.cell_mapping(cell,self.param.n_x)
      i_mat = self.param.mat_id[i,j]
      sca = self.param.sig_s[:,i_mat]
# Get location in the matrix
      ii = t_s.mapping(i,j,self.param.n_x)
# Block diagonal term
      for k in xrange(0,self.param.n_mom) :
        kk = k*4*self.param.n_cells + ii
        tmp = x[kk[0]:kk[3]+1]
        dot_product = np.dot(self.fe.mass_matrix,tmp)
        pos = 0
        for i_kk in xrange(int(kk[0]),int(kk[3]+1)) :
          self.scattering_src[i_kk] += self.scattering_src[i_kk] + sca[k]*\
              dot_product[pos]
          pos += 1

#----------------------------------------------------------------------------#

  def sweep(self,color) :
    """Perform the colored sweep on all the directions. The function is purely
    virtual."""

    raise NotImplementedError("sweep is purely virtual and must be overridden.")

#----------------------------------------------------------------------------#

  def update_flux_moments(self,flux_moments) :
    """Update self.flux_moments after a bloc sweep."""

    for i in xrange(0,self.flux_moments.shape[0]) :
      if flux_moments[i]!=0 :
        self.flux_moments[i] = flux_moments[i]

#----------------------------------------------------------------------------#

  def update_angular_flux(self,psi_ij) :
    """Update self.all_psi after a bloc sweep."""

    for idir in xrange(0,self.quad.n_dir) :
      for i in xrange(0,4*self.param.n_cells) :
        if psi_ij[idir,i]!=0 :
          self.all_psi[idir][i] = psi_ij[idir,i]
