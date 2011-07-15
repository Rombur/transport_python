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

  def sweep(self) :
    """Do the transport sweep on all the directions."""

    raise NotImplementedError("sweep is purely virtual and must be\
        overriden.")
