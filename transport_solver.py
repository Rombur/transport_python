# Python code
# Author: Bruno Turcksin
# Date: 2011-04-01 10:15:51.682471

#----------------------------------------------------------------------------#
## Class transport_solver                                                   ##
#----------------------------------------------------------------------------#

"""Contain the solver of the transport equation"""

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
import parameters
import quadrature
import finite_element

class transport_solver(object) :
  """Solve the transport equation in 2D for cartesian geometry"""

  def __init__(self,param,tol,max_it,output_file,mv_time=None,
      mip_time=None) :

    super(transport_solver,self).__init__()
    self.tol = tol
    self.max_iter = max_it
    self.param = param
# Create and build the quadrature
# When the transport solver is reduced to the MIP or the P1SA. There is no
# quadrature, we don't need the quadrature which sometimes does not exist.
    if np.fmod(self.param.sn,2)==0 :
      self.quad = quadrature.quadrature(param)
      self.quad.build_quadrature()
# Compute the most normal direction
    if np.fmod(self.param.sn,2)==0 :
       self.most_normal()
# Save the values to be printed in a file 
    self.output_file = output_file

#----------------------------------------------------------------------------#

  def most_normal(self) :
    """Compute the most normal direction among the quadrature in the
    quadrature set."""

    max_elements = self.quad.omega.max(0) 
    min_elements = self.quad.omega.min(0) 
    
    left = []
    right = []
    top = []
    bottom = []
    for i in xrange(0,self.quad.n_dir) :
      if self.quad.omega[i,0] == max_elements[0] :
        left.append(i)
      elif self.quad.omega[i,0] == min_elements[0] :
        right.append(i)
      elif self.quad.omega[i,1] == max_elements[1] :
        bottom.append(i)
      elif self.quad.omega[i,1] == min_elements[1] :
        top.append(i)

    self.most_n = {'left' : left, 'right' : right, 'top' : top,\
        'bottom' : bottom}

#----------------------------------------------------------------------------#

  def count_gmres_iterations(self,residual) :
    """Callback function called at the end of each GMRES iteration. Count the
    number of iterations and compute the L2 norm of the current residual."""

    self.gmres_iteration += 1
    res = scipy.linalg.norm(residual) 
    if self.param.verbose>0 :
      self.print_message('L2-norm of the residual for iteration %i'\
          %self.gmres_iteration+' : %f'%scipy.linalg.norm(residual))

#----------------------------------------------------------------------------#

  def cell_mapping(self,cell) :
    """Get the i,j pair of a cell given a cell."""

    j = np.floor(cell/self.param.n_x)
    i = cell - j*self.param.n_x

    return i,j

#----------------------------------------------------------------------------#

  def mapping(self,i,j) :
    """Compute the index in the 'local' matrix."""

    cell = int(i+j*self.param.n_x)
    index = range(4*cell,4*(cell+1))

    return index

#----------------------------------------------------------------------------#

  def sweep(self,gmres_rhs) :
    """Do the transport sweep on all the directions."""
                                                    
    tmp = int(4*self.param.n_cells)
    self.scattering_src = self.scattering_src.reshape(self.param.n_mom,tmp)

    [x_down,x_up,y_down,y_up] = self.upwind_edge()
    
    flux_moments = np.zeros((4*self.param.n_mom*self.param.n_cells))
    for idir in xrange(0,self.quad.n_dir) :
      psi = np.zeros((4*self.param.n_cells))

# Direction alias
      omega_x = self.quad.omega[idir,0]
      omega_y = self.quad.omega[idir,1]

# Upwind/downwind indices
      if omega_x > 0.0 :
        sx = 0
        x_begin = 0
        x_end = int(self.param.n_x)
        x_incr = 1
      else :
        sx = 1
        x_begin = int(self.param.n_x-1)
        x_end = -1
        x_incr = -1
      if omega_y > 0.0 :
        sy = 0
        y_begin = 0
        y_end = int(self.param.n_y)
        y_incr = 1
      else :
        sy = 1
        y_begin = int(self.param.n_y-1)
        y_end = -1
        y_incr = -1

# Compute the gradient 
      gradient = omega_x*(-self.fe.x_grad_matrix+x_down[sx,:,:])+omega_y*(
          -self.fe.y_grad_matrix+y_down[sy,:,:])
      
      for i in xrange(x_begin,x_end,x_incr) :
        for j in xrange(y_begin,y_end,y_incr) :
          i_mat = self.param.mat_id[i,j]
          sig_t = self.param.sig_t[i_mat]

# Volumetric term of the rhs
          i_src = self.param.src_id[i,j]
          rhs = np.zeros((4))
          if gmres_rhs == True :
            rhs = self.param.src[i_src]*self.fe.width_cell[0]*\
                self.fe.width_cell[1]*np.ones((4))/4.

# Get location in the matrix
          ii = self.mapping(i,j)

# Add scattering source contribution
          scat_src = np.dot(self.quad.M[idir,:],self.scattering_src[:,ii])
          rhs += scat_src

# Block diagonal term
          L = gradient+sig_t*self.fe.mass_matrix

# Upwind term in x
          if i>0 and sx==0 :
            jj = self.mapping(i-1,j)
            rhs -= omega_x*np.dot(x_up[sx,:,:],psi[jj])
          else :
            if i==0 and idir in self.most_n['left'] and gmres_rhs==True :
              rhs -= omega_x*np.dot(x_up[sx,:,:],self.param.inc_left[j]*\
                  np.ones((4)))
          if i<self.param.n_x-1 and sx==1 :
            jj = self.mapping(i+1,j)
            rhs -= omega_x*np.dot(x_up[sx,:,:],psi[jj])
          else :
            if i==self.param.n_x-1 and idir in self.most_n['right'] and\
                gmres_rhs==True :
              rhs -= omega_x*np.dot(x_up[sx,:,:],self.param.inc_right[j]*\
                  np.ones((4)))

          if j>0 and sy==0 :
            jj = self.mapping(i,j-1)
            rhs -= omega_y*np.dot(y_up[sy,:,:],psi[jj])
          else :
            if j==0 and idir in self.most_n['bottom'] and gmres_rhs==True :
              rhs -= omega_y*np.dot(y_up[sy,:,:],self.param.inc_bottom[i]*\
                  np.ones((4)))
          if j<self.param.n_y-1 and sy==1 :
            jj = self.mapping(i,j+1)
            rhs -= omega_y*np.dot(y_up[sy,:,:],psi[jj])
          else :
            if j==self.param.n_y-1 and idir in self.most_n['top'] and\
                gmres_rhs==True :
              rhs -= omega_y*np.dot(y_up[sy,:,:],self.param.inc_top[i]*\
                  np.ones((4)))

          psi[ii] = scipy.linalg.solve(L,rhs,sym_pos=False,lower=False,
            overwrite_a=True,overwrite_b=True)

# update scalar flux
      for i in xrange(0,self.param.n_mom) :
        i_begin = i*4*self.param.n_cells
        i_end = (i+1)*4*self.param.n_cells
        flux_moments[i_begin:i_end] += self.quad.D[i,idir]*psi[:]
        
    
    return flux_moments

#----------------------------------------------------------------------------#

  def project_vector(self,x) :
    """Project a vector from coarse_n_mom to self.n_mom."""

    n_dofs = 4*self.param.n_cells
    projection = np.zeros(4*self.param.n_mom*self.param.n_cells)
    coarse_n_mom = x.shape[0]/(4*self.param.n_cells)
    if self.param.galerkin==True :
      skip = (-1+np.sqrt(1+2*self.param.n_mom))/2-1
      tmp_end = coarse_n_mom-(skip-1)
      residual = coarse_n_mom-tmp_end
      projection[0:n_dofs*tmp_end] = x[0:n_dofs*tmp_end]
      projection[n_dofs*(tmp_end+skip):n_dofs*(tmp_end+skip+residual)] =\
          x[n_dofs*tmp_end:n_dofs*(tmp_end+residual)]
    else :
      projection[0:4*self.n_mom*self.param.n_cells] = x[0:4*coarse_n_mom*\
          self.param.n_cells]
    
    return projection  

#----------------------------------------------------------------------------#

  def restrict_vector(self,x) :
    """Project a vector from refine_n_mom to self.n_mom."""

    n_dofs = 4*self.param.n_cells
    restriction= np.zeros(4*self.param.n_mom*self.param.n_cells)
    refine_n_mom = x.shape[0]/(4*self.param.n_cells)
    if self.param.galerkin==True :
      skip = self.param.sn-1
      tmp_end = self.param.n_mom-(skip-1)
      residual = self.param.n_mom-tmp_end
      restriction[0:n_dofs*tmp_end] = x[0:n_dofs*tmp_end]
      restriction[n_dofs*tmp_end:n_dofs*(tmp_end+residual)] =\
          x[n_dofs*(tmp_end+skip):n_dofs*(tmp_end+skip+residual)]
    else :
      restriction[0:4*self.n_mom*self.param.n_cells] = x[0:4*self.n_mom*\
          self.param.n_cells]
    
    return restriction  

#----------------------------------------------------------------------------#

  def print_message(self,a) :
    """Print the given message a on the screen or in a file."""

    if self.param.print_to_file==True :
      self.output_file.write(a+'\n')
    else :
      print a 

#----------------------------------------------------------------------------#

  def solve(self) :
    """Solve the transport equation. The function is purely virtual."""

    raise NotImplementedError("solve is purely virtual and must be overriden.")

#----------------------------------------------------------------------------#

  def mv(self,x) :
    """Perform the matrix-vector multiplication needed by GMRES. The function
    is purely virtual."""

    raise NotImplementedError("solve is purely virtual and must be overriden.")

#----------------------------------------------------------------------------#

  def compute_scattering_source(self,x) :
    """Compute the scattering source given a flux. The function is purely
    virtual."""

    raise NotImplementedError("compute_scattering_source is purely virtual\
        and must be overriden.")

#----------------------------------------------------------------------------#

  def upwind_edge(self) :
    """Compute the edge for the upwind. The function is purely virtual."""

    raise NotImplementedError("upwind_edge is purely virtual and must be\
        overriden.")
