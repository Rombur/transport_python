# Python code
# Author: Bruno Turcksin
# Date: 2011-04-01 10:15:51.682471

#----------------------------------------------------------------------------#
## Class TRANSPORT_SOLVER                                                   ##
#----------------------------------------------------------------------------#

"""Contain the solver of the transport equation"""

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
import DOF_HANDLER
import FINITE_ELEMENT
import PARAMETERS
import QUADRATURE

class TRANSPORT_SOLVER(object) :
  """Solve the transport equation in 2D for cartesian geometry"""

  def __init__(self,param,tol,max_it,output_file) :

    super(TRANSPORT_SOLVER,self).__init__()
    self.tol = tol
    self.max_iter = max_it
    self.param = param
# Create and build the quadrature
    self.quad = QUADRATURE.QUADRATURE(param)
    self.quad.Build_quadrature()
# Compute the most normal direction
    self.Most_normal()
# Build the dof_handler and compute the sweep ordering
    self.dof_handler = DOF_HANDLER.DOF_HANDLER(param)
    self.dof_handler.Compute_sweep_ordering(self.quad,param)
# Save the values to be printed in a file 
    self.output_file = output_file

#----------------------------------------------------------------------------#

  def Most_normal(self) :
    """Compute the most normal direction among the quadrature in the
    quadrature set."""

    max_elements = self.quad.omega.max(0) 
    min_elements = self.quad.omega.min(0) 
    
    left = []
    right = []
    top = []
    bottom = []
    for i in range(0,self.quad.n_dir) :
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

  def Count_gmres_iterations(self,residual) :
    """Callback function called at the end of each GMRES iteration. Count the
    number of iterations and compute the L2 norm of the current residual."""

    self.gmres_iteration += 1
    res = scipy.linalg.norm(residual) 
    if self.param.verbose>0 :
      self.Print_message('L2-norm of the residual for iteration %i'\
          %self.gmres_iteration+' : %f'%scipy.linalg.norm(residual))
#----------------------------------------------------------------------------#

  def Print_message(self,message) :
    """Print the given message on the screen or in a file."""

    if self.param.print_to_file==True :
      self.output_file.write(a+'\n')
    else :
      print(message)

#----------------------------------------------------------------------------#

  def Solve(self) :
    """Solve the transport equation."""

    start = time.time()
    if self.param.gmres==True :
# Compute the uncollided flux moment (D*inv(L)*q) = rhs of gmres
      gmres_rhs = True
      self.scattering_src = np.zeros((self.param.n_mom,4*self.param.n_cells))
      self.gmres_b = self.Sweep(gmres_rhs)

# GMRES solver 
      self.gmres_iteration = 0 
      size = self.gmres_b.shape[0]
      A = scipy.sparse.linalg.LinearOperator((size,size),matvec=self.Mv,
          rmatvec=None,dtype=float)
      self.flux_moments,flag = scipy.sparse.linalg.gmres(A,self.gmres_b,
          x0=None,tol=self.tol,restrt=20,maxiter=self.max_iter,M=None,
          callback=self.Count_gmres_iterations)

      if flag!=0 :
        self.Print_message('Transport did not converge.')

    else :
      rhs = True
      self.flux_moments = np.zeros((4*self.param.n_cells*self.param.n_mom))
      flux_moments_old = np.zeros((4*self.param.n_cells*self.param.n_mom))
      for i in range(0,self.max_iter) :
        self.compute_scattering_source(flux_moments_old)
        self.flux_moments = self.sweep(rhs)

        conv = scipy.linalg.norm(self.flux_moments-flux_moments_old)/\
            scipy.linalg.norm(self.flux_moments)
        
        self.Print_message('L2-norm of (phi^k+1 - phi^k)/phi^k for iteration %i'\
            %i+' : %f'%conv)
        if conv<self.tol :
          break

        flux_moments_old = self.flux_moments.copy()
        
    end = time.time()

    self.Print_message('Elapsed time to solve the problem : %f'%(end-start))

#----------------------------------------------------------------------------#

  def Mv(self,x) :
    """Perform the matrix-vector multiplication needed by GMRES."""

    y=x.copy()

# Compute the scattering source
    self.Compute_scattering_source(y)

# Do a transport sweep (no iteration on significant angular fluxes, we assume
# that no BC are reflective)
    flxm = self.Sweep(False)
    sol = y-flxm

    return sol

#----------------------------------------------------------------------------#

  def Compute_scattering_source(self,x) :
    """Compute the scattering source given a flux. The function is purely
    virtual."""

    raise NotImplementedError("compute_scattering_source is purely virtual\
        and must be overriden.")

#----------------------------------------------------------------------------#

  def Sweep(self,gmres_rhs) :
    """Do the transport sweep on all the directions."""

    raise NotImplementedError("sweep is purely virtual and must be\
        overriden.")
