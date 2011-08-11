# Python code
# Author: Bruno Turcksin
# Date: 2011-07-15 10:21:09.909533

#----------------------------------------------------------------------------#
## Class rbow_block_transport_solver                                        ##
#----------------------------------------------------------------------------#

"""Contain the red-black-orange-white block solver of the transport equation."""
import numpy as np
import scipy.linalg
import transport_solver as t_s
import block_transport_solver as b_t_s

class rbow_block_transport_solver(b_t_s.block_transport_solver) :
  """Solve the transport equation in 2D for cartesian geometry using a
  red-black-orange-white block algorithm. Works only with SI and if n_x and
  n_y are even. Derived from block_transport_solver."""

  def __init__(self,param,fe,quad,flux_moments,all_psi,x_down,x_up,y_down,\
      y_up,most_normal) :

    super(rbow_block_transport_solver,self).__init__(param,fe,quad,flux_moments,\
        all_psi,x_down,x_up,y_down,y_up,most_normal)

#----------------------------------------------------------------------------#

  def solve(self) :
    """Driver for the red-black-orange-white block sweep."""

    self.compute_scattering_source(self.flux_moments)
    flux_moments = self.sweep('red')
    self.update_flux_moments(flux_moments)
    self.compute_scattering_source(self.flux_moments)
    flux_moments = self.sweep('black')
    self.update_flux_moments(flux_moments)
    self.compute_scattering_source(self.flux_moments)
    flux_moments = self.sweep('orange')
    self.update_flux_moments(flux_moments)
    self.compute_scattering_source(self.flux_moments)
    flux_moments = self.sweep('white')
    self.update_flux_moments(flux_moments)

    return self.flux_moments

#----------------------------------------------------------------------------#

  def sweep(self,color) :
    """Perform the red-black-orange-white sweep on all the directions."""

    tmp = int(4*self.param.n_cells)
    self.scattering_src = self.scattering_src.reshape(self.param.n_mom,tmp)

    flux_moments = np.zeros((4*self.param.n_mom*self.param.n_cells))
    psi_ij = np.zeros((self.quad.n_dir,4*self.param.n_cells))
    
    if color=='red' :
      i_begin = 0
      i_end = self.param.n_x-1
      j_begin = 0
      j_end = self.param.n_y-1
    elif color=='black' :
      i_begin = 1
      i_end = self.param.n_x-2
      j_begin = 0
      j_end = self.param.n_y-1
    elif color=='orange' :
      i_begin = 0
      i_end = self.param.n_x-1
      j_begin = 1
      j_end = self.param.n_y-2
    else :
      i_begin = 1
      i_end = self.param.n_x-2
      j_begin = 1
      j_end = self.param.n_y-2
    
    for i in xrange(i_begin,i_end,2) :
      for j in xrange(j_begin,j_end,2) :
        for idir in xrange(0,self.quad.n_dir) :
          psi = np.zeros((4*self.param.n_cells))

# Direction alias
          omega_x = self.quad.omega[idir,0]
          omega_y = self.quad.omega[idir,1]

# Upwind/downwind indices 
          if omega_x>0.0 :
            sx = 0
            x_begin = i
            x_end = i+2
            x_incr = 1
          else :
            sx = 1
            x_begin = i+1
            x_end = i-1
            x_incr = -1
          if omega_y>0.0 :
            sy = 0
            y_begin = j
            y_end = j+2
            y_incr = 1
          else :
            sy = 1
            y_begin = j+1
            y_end = j-1
            y_incr = -1

# Compute the gradient
          gradient = omega_x*(-self.fe.x_grad_matrix+self.x_down[sx,:,:])+\
              omega_y*(-self.fe.y_grad_matrix+self.y_down[sy,:,:])

          for m in xrange(x_begin,x_end,x_incr) :
            for n in xrange(y_begin,y_end,y_incr) : 
              i_mat = self.param.mat_id[i,j]
              sig_t = self.param.sig_t[i_mat]

# Volumetric term of the rhs
              i_src = self.param.src_id[m,n]
              rhs = self.param.src[i_src]*self.fe.width_cell[0]*\
                  self.fe.width_cell[1]*np.ones((4))/4.

# Get location in the matrix
              ii = t_s.mapping(m,n,self.param.n_x)

# Add scattering source contribution 
              scat_src = np.dot(self.quad.M[idir,:],self.scattering_src[:,ii])
              rhs += scat_src

# Block diagonal term
              L = gradient+sig_t*self.fe.mass_matrix

# Upwind term in x
              if m>0 and sx==0 :
                jj = t_s.mapping(m-1,n,self.param.n_x)
                if m==x_begin :
                  rhs -= omega_x*np.dot(self.x_up[sx,:,:],\
                      self.all_psi[idir][jj])
                else :
                  rhs -= omega_x*np.dot(self.x_up[sx,:,:],psi[jj])
              elif m==0 and idir in self.most_n['left'] :
                rhs -= omega_x*np.dot(self.x_up[sx,:,:],\
                    self.param.inc_left[n]*np.ones((4)))
              if m<self.param.n_x-1 and sx==1 :
                jj = t_s.mapping(m+1,n,self.param.n_x)
                if m==x_begin :
                  rhs -=omega_x*np.dot(self.x_up[sx,:,:],\
                      self.all_psi[idir][jj])
                else :
                  rhs -= omega_x*np.dot(self.x_up[sx,:,:],psi[jj])
              elif m==self.param.n_x-1 and idir in self.most_n['right'] :
                rhs -= omega_x*np.dot(self.x_up[sx,:,:],\
                    self.param.inc_right[j]*np.ones((4)))

# Upwind term in y
              if n>0 and sy==0 :
                jj = t_s.mapping(m,n-1,self.param.n_x)
                if n==y_begin :
                  rhs -= omega_y*np.dot(self.y_up[sy,:,:],\
                      self.all_psi[idir][jj])
                else :
                  rhs -= omega_y*np.dot(self.y_up[sy,:,:],psi[jj])
              elif n==0 and idir in self.most_n['bottom'] :
                rhs -= omega_y*np.dot(self.y_up[sy,:,:],\
                    self.param.inc_bottom[i]*np.ones((4)))
              if n<self.param.n_y-1 and sy==1 :
                jj = t_s.mapping(m,n+1,self.param.n_x)
                if n==y_begin :
                  rhs -= omega_y*np.dot(self.y_up[sy,:,:],psi[jj])
                else :
                  rhs -= omega_y*np.dot(self.y_up[sy,:,:],psi[jj])
              elif n==self.param.n_y-1 and idir in self.most_n['top'] :
                rhs -= omega_y*np.dot(self.y_up[sy,:,:],\
                    self.param.inc_top[i]*np.ones((4)))

              psi[ii] = scipy.linalg.solve(L,rhs,sym_pos=False,lower=False,
                  overwrite_a=True,overwrite_b=True)

              ratio = .25
              if i==0 or i==self.param.n_x-2 :
                ratio *= 2.
              if j==0 or j==self.param.n_y-2 :
                ratio *= 2.
            
              psi_ij[idir][ii] += ratio*psi[ii]

# Update the flux_moments
    for k in xrange(0,self.param.n_mom) :
      k_begin = k*4*self.param.n_cells
      k_end = (k+1)*4*self.param.n_cells
      for idir in xrange(0,self.quad.n_dir) :
        flux_moments[k_begin:k_end] += self.quad.D[k,idir]*psi_ij[idir,:]
            
# Update the angular flux
    self.update_angular_flux(psi_ij)

    return flux_moments
