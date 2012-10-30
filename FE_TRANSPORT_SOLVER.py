# Python code
# Author: Bruno Turcksin
# Date: 2011-06-08 12:28:45.930931

#----------------------------------------------------------------------------#
## Class FE_TRANSPORT_SOLVER                                                ##
#----------------------------------------------------------------------------#

"""Contain the finite elements solver of the transport equation"""

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
import TRANSPORT_SOLVER as SOLVER

class FE_TRANSPORT_SOLVER(SOLVER.TRANSPORT_SOLVER)  :
  """Solve the transport equation discretized with finite elements."""

  def __init__(self,param,tol,max_it,output_file) :

    super(FE_TRANSPORT_SOLVER,self).__init__(param,tol,max_it,output_file)

#----------------------------------------------------------------------------#

  def Compute_scattering_source(self,x) :
    """Compute the scattering source given a flux."""

    self.scattering_src = np.zeros((self.param.n_mom,self.dof_handler.n_dof))
    for cell in self.dof_handler.mesh :
      for k in xrange(0,self.param.n_mom) :
        offset = k*self.dof_handler.n_dof
        tmp = x[offset+cell.first_dof:offset+cell.last_dof]
        dot_product = np.dot(self.fe.mass_matrix,tmp)
        for i in xrange(cell.first_dof,cell.last_dof) :
          self.scattering[k,i] += cell.sigma_s[k]*tmp[i-cell.first_dof]

#----------------------------------------------------------------------------#

  def Sweep(self,gmres_rhs) :
    """Do the transport sweep on all the directions."""
                                                    
    flux_moments = np.zeros((self.param.n_mom*self.dof_handler.n_dof))
    for idir in xrange(0,self.quad.n_dir) :
      psi = np.zeros((self.fe.n_dof_per_cell*self.param.n_cells))

# Direction alias
      omega_x = self.quad.omega[idir,0]
      omega_y = self.quad.omega[idir,1]

      for i in self.dof_handler.sweep_order[idir] :
        cell = self.dof_handler.mesh[i]
# Compute the volume term 
        L = -omega_x*cell.sd.x_grad_matrix-omega_y*\
            cell.sd.y_grad_matrix+cell.sigma_t*cell.sd.mass_matrix

# Volumetric term of the rhs
        rhs = np.zeros((4))
        if gmres_rhs==True :
          rhs = cell.src*cell.width[0]*cell.width[1]*np.ones((4))/4.

# Add scattering source contribution
        scat_src = np.dot(self.quad.M[idir,:],
            self.scattering_src[:,cell.first_dof:cell.last_dof])
        rhs += scat_src

        if omega_x>0. :
          if omega_y>0. :
# Upwind terms
            if cell.v0[0]==self.dof_handler.left :
              if idir in self.most_n['left'] :
                inc_flux = self.param.inc_left/self.param.weight*np.ones((4))
                rhs -= omega_x*np.dot(self.upwind[3],inc_flux)
            if cell.v0[1]==self.dof_handler.bottom :
              if idir in self.most_n['bottom'] :
                inc_flux = self.param.inc_bottom/self.param.weight*np.ones((4))
                rhs -= omega_y*np.dot(self.dof_handler[0],inc_flux)
# Downwind terms
            L += omega_x*self.downwind[1]+omega_y*self.downwind[2]
          else :
# Upwind terms
            if cell.v0[0]==self.dof_handler.left :
              if idir in self.most_n['left'] :
                inc_flux = self.param.inc_left/self.param.weight*np.ones((4))
                rhs -= omega_x*np.dot(self.upwind[3],inc_flux)
            if cell.v3[1]==self.dof_handler.up :
              if idir in self.most_n['top'] :
                inc_flux = self.param.inc_top/self.param.weight*np.ones((4))
                rhs -= omega_y*np.dot(self.upwind[2],inc_flux)
# Downwind terms
            L += omega_x*self.downwind[1]+omega_y*self.downwind[0]    
        else :
          if omega_y>0. :
# Upwind terms
            if cell.v1[0]==self.dof_handler.right :
              if idir in self.most_n['right'] :
                inc_flux = self.param.inc_right/self.param.weight*np.ones((4))
                rhs -= omega_x*np.dot(self.upwind[1],inc_flux)
            if cell.v1[1]==self.dof_handler.bottom :
              if idir in self.most_n['bottom'] :
                inc_flux = self.param.inc_bottom/self.param.weight*np.ones((4))
                rhs -= omega_y*np.dot(self.upwind[0],inc_flux)
# Downwind terms
            L += omega_x*self.downwind[3]+omega_y*self.downwind[2]
          else :
# Upwind terms
            if cell.v1[0]==self.dof_handler.right :
              if idir in self.most_n['rigth'] :
                inc_flux = self.param.inc_right/self.param.weight*np.ones((4))
                rhs -= omega_x*np.dot(self.upwind[1],inc_flux)
            if cell.v2[1]==self.dof_handler.top :
              if idir in self.most_n['top'] :
                inc_flux = self.param.inc_top/self.param.weight*np.ones((4))
                rhs -= omega_y*np.dot(self.upwind[2],inc_flux)
# Downwind terms
            L += omega_x*self.downwin[3]+omega_y*self.downwind[0]

        psi[cell.first_dof:cell.last_dof] = scipy.linalg.solve(L,rhs,sym_pos=False,lower=False,
            overwrite_a=True,overwrite_b=True)

# Update scalar flux
      for i in xrange(0,self.param.n_mom) :
        i_begin = i*self.dof_handler.n_dof
        i_end = (i+1)*self.dof_handler.n_dof
        flux_moments[i_begin:i_end] += self.quad.D[i,idir]*psi[:]
        
    return flux_moments
