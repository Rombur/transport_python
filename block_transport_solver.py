# Python code
# Author: Bruno Turcksin
# Date: 2011-07-10 18:57:42.034268

#----------------------------------------------------------------------------#
## Class block_transport_solver                                             ##
#----------------------------------------------------------------------------#

"""Contain the block solver of the transport equation"""

import numpy as np
import scipy.linalg
import transport_solver as t_s

class block_transport_solver(object) :
  """Solve the transport equation in 2D for cartesian geometry, using a
  red-black algorithm. Works only with SI."""

  def __init__(self,param,fe,quad,flux_moments,all_psi,x_down,x_up,y_down,\
      y_up,most_normal,tol,max_it) :

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
    self.tol = tol
    self.max_iter = max_it

#----------------------------------------------------------------------------#

  def solve(self) :
    """Driver for the red-black block sweep."""

    self.compute_scattering_source(self.flux_moments)
    flux_moments = self.sweep('red')
    self.update_flux_moments(flux_moments)
    self.compute_scattering_source(self.flux_moments)
    flux_moments = self.sweep('black')
    self.update_flux_moments(flux_moments)

    return self.flux_moments

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
    """Do the red and the black sweep on all the directions."""

    tmp = int(4*self.param.n_cells)
    self.scattering_src = self.scattering_src.reshape(self.param.n_mom,tmp)

    flux_moments = np.zeros((4*self.param.n_mom*self.param.n_cells))
    psi_ij = np.zeros((self.quad.n_dir,4*self.param.n_cells))
    for i in xrange(0,self.param.n_x) :
      for j in xrange(0,self.param.n_y) :
        if (np.mod(i+j,2)==0 and color=='red') or (np.mod(i+j,2)==1 and\
            color=='black') :
          for idir in xrange(0,self.quad.n_dir) :
            psi = np.zeros((4*self.param.n_cells))

            x_begin = []
            x_end = []
            y_begin = []
            y_end = []

# Direction alias
            omega_x = self.quad.omega[idir,0]
            omega_y = self.quad.omega[idir,1]

# Upwind/downwind indices
            if omega_x > 0.0 :
              sx = 0
# Left
              if i!=0 :
                x_begin.append(i-1)
                x_end.append(i)
# Right
              if i!=int(self.param.n_x-1) :
                x_begin.append(i)
                x_end.append(i+1)
# Bottom 
              if j!=0 :
                x_begin.append(i)
                x_end.append('')
# Top
              if j!=int(self.param.n_y-1) :
                x_begin.append(i)
                x_end.append('')
            else :
              sx = 1
# Left
              if i!=0 :
                x_begin.append(i)
                x_end.append(i-1)
              
              if i!=int(self.param.n_x-1) :
                x_begin.append(i+1)
                x_end.append(i)

              if j!=0 :
                x_begin.append(i)
                x_end.append('')

              if j!=int(self.param.n_y-1) :
                x_begin.append(i)
                x_end.append('')

            if omega_y > 0.0 :
              sy = 0
              
              if i!=0 :
                y_begin.append(j)
                y_end.append('')

              if i!=int(self.param.n_x-1) :
                y_begin.append(j)
                y_end.append('')

              if j!=0 :
                y_begin.append(j-1)
                y_end.append(j)

              if j!=int(self.param.n_y-1) :
                y_begin.append(j)
                y_end.append(j+1)

            else :
              sy = 1

              if i!=0 :
                y_begin.append(j)
                y_end.append('')
              
              if i!=int(self.param.n_x-1) :
                y_begin.append(j)
                y_end.append('')
              
              if j!=0 :
                y_begin.append(j)
                y_end.append(j-1)

              if j!=int(self.param.n_y-1) :
                y_begin.append(j+1)
                y_end.append(j)

# Compute the gradient 
            gradient = omega_x*(-self.fe.x_grad_matrix+self.x_down[sx,:,:])+\
                omega_y*(-self.fe.y_grad_matrix+self.y_down[sy,:,:])
              
            for k in xrange(0,len(x_begin)) :
#              psi = np.zeros((4))
              x = []
              y = []
              x.append(x_begin[k])
              y.append(y_begin[k])
              if x_end[k]!='' :
                x.append(x_end[k])
              if y_end[k]!='' :
                y.append(y_end[k])
              for m in x :
                for n in y :
                  i_mat = self.param.mat_id[m,n]
                  sig_t = self.param.sig_t[i_mat]

# Volumetric term of the rhs
                  i_src = self.param.src_id[i,j]
                  rhs = self.param.src[i_src]*self.fe.width_cell[0]*\
                      self.fe.width_cell[1]*np.ones((4))/4.
                     
# Get location in the matrix
                  ii = t_s.mapping(m,n,self.param.n_x)

# Add scattering source constribution
                  scat_src = np.dot(self.quad.M[idir,:],self.scattering_src[:,ii])
                  rhs += scat_src

# Block diagonal term     
                  L = gradient+sig_t*self.fe.mass_matrix

# Upwind term in x
                  if m>0 and sx==0 :
                    jj = t_s.mapping(m-1,n,self.param.n_x)
                    if m==x_begin[k] :
                      rhs -= omega_x*np.dot(self.x_up[sx,:,:],\
                          self.all_psi[idir][jj])
                    else :
                      rhs -= omega_x*np.dot(self.x_up[sx,:,:],psi[jj])
                  elif m==0 and idir in self.most_n['left'] :
                    rhs -= omega_x*np.dot(self.x_up[sx,:,:],\
                        self.param.inc_left[n]*np.ones((4)))
                  if m<self.param.n_x-1 and sx==1 :
                    jj = t_s.mapping(m+1,n,self.param.n_x)
                    if m==x_begin[k] :
                      rhs -= omega_x*np.dot(self.x_up[sx,:,:],\
                          self.all_psi[idir][jj])
                    else :
                      rhs -= omega_x*np.dot(self.x_up[sx,:,:],psi[jj])
                  elif m==self.param.n_x-1 and idir in self.most_n['right'] :
                    rhs -= omega_x*np.dot(self.x_up[sx,:,:],\
                        self.param.inc_right[j]*np.ones((4)))

# Upwind term in y
                  if n>0 and sy==0 :
                    jj = t_s.mapping(m,n-1,self.param.n_x)
                    if n==y_begin[k] :
                      rhs -= omega_y*np.dot(self.y_up[sy,:,:],\
                          self.all_psi[idir][jj])
                    else :
                      rhs -= omega_y*np.dot(self.y_up[sy,:,:],psi[jj])
                  elif n==0 and idir in self.most_n['bottom'] :
                    rhs -= omega_y*np.dot(self.y_up[sy,:,:],\
                        self.param.inc_bottom[i]*np.ones((4)))
                  if n<self.param.n_y-1 and sy==1 :
                    jj = t_s.mapping(m,n+1,self.param.n_x)
                    if n==y_begin[k] :
                      rhs -= omega_y*np.dot(self.y_up[sy,:,:],\
                          self.all_psi[idir][jj])
                    else :
                      rhs -= omega_y*np.dot(self.y_up[sy,:,:],psi[jj])
                  elif n==self.param.n_y-1 and idir in self.most_n['top'] :
                    rhs -= omega_y*np.dot(self.y_up[sy,:,:],\
                        self.param.inc_top[i]*np.ones((4)))
                  
                  psi[ii] = scipy.linalg.solve(L,rhs,sym_pos=False,lower=False,
                    overwrite_a=True,overwrite_b=True)

                  if m==i and n==j :
                    psi_ij[idir][ii] += psi[ii]/len(x_begin)

# Update the flux_moments
    for k in xrange(0,self.param.n_mom) :
      k_begin = k*4*self.param.n_cells
      k_end = (k+1)*4*self.param.n_cells
      for idir in xrange(0,self.quad.n_dir) :
        flux_moments[k_begin:k_end] += self.quad.D[k,idir]*psi_ij[idir,:]
            
# Update the angular flux
    self.update_angular_flux(psi_ij)

    return flux_moments

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
