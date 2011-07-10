# Python code
# Author: Bruno Turcksin
# Date: 2011-07-04 15:32:11.045280

#----------------------------------------------------------------------------#
## Class SV_transport_solver                                                ##
#----------------------------------------------------------------------------#

"""Contain the spectral volumes solver of the transport equation"""

import numpy as np
import time 
import transport_solver as solver
import spectral_volume

class SV_transport_solver(solver.tranport_solver) :
  """Solve the transport equation discretized with spectral volumes."""

  def __init__(self,param,tol,max_it,output_file,point_value) :

    super(SV_transport_solver,self).__init__(param,tol,max_it,output_file)

# Create and build the spectral volume object 
    self.sv = spectral_volume.spectral_volume(param)
    self.sv.build_edge_integral()
    self.sv.build_cell_integral(point_value)

#----------------------------------------------------------------------------#

  def solve(self) :
    """Solve the transport equation."""

    start = time.time()
    if self.param.gmres==True :
# Compute the uncollided flux moment (D*inv(L)*q) = rhs of gmres
      gmres_rhs = True
      self.scattering_src = np.zeros((self.parm.n_mom,4*self.param.n_cells))
      self.gmres_b = self.sweep(gmres_rhs)

# GMRES solver 
      self.gmres_iteration = 0 
      size = self.gmres_b.shape[0]
      A = scipy.sparse.linalg.LinearOperator((size,size),matvec=self.mv,
          rmatvec=None,dtype=float)
      self.flux_moments,flaf = scipy.sparse.linalg.gmres(A,self.gmres_b,
          x0=None,tol=self.tol,restrt=20,maxiter=self.max_iter,M=None,
          callback=self.count_gmres_iterations)

      if flag!=0 :
        self.print_message('Transport did not converge.')

    else :
      rhs = True
      self.flux_moments = np.zeros((4*self.param.n_cells*self.param.n_mom))
      flux_moments_old = np.zeros((4*self.param.n_cells*self.param.n_mom))
      for i in xrange(0,self.max_iter) :
        self.compute_scattering_source(flux_moments_old)
        self.flux_moments = self.sweep(rhs)

        conv = scipy.linalg.norm(self.flux_moments-flux_moments_old)/\
            scipy.linalg.norm(self.flux_moments)
        
        self.print_message('L2-norm of the residual for iteration %i'\
            %i+' : %f'%conv)
        if conv<self.tol :
          break

        flux_moments_old = self.flux_moments.copy()
        
    end = time.time()

    self.print_message('Elapsed time to solve the problem : %f'%(end-start))

#----------------------------------------------------------------------------#

  def mv(self,x) :
    """Perform the matrix-vector multiplication needed by GMRES."""

    mv_start = time.time()
    y=x.copy()

# Compute the scattering source
    self.compute_scattering_source(y)

# Do a transport sweep (no iteration on significant angular fluxes, we assume
# that no BC are reflective)
    flxm = self.sweep(False)
    sol = y-flxm

    mv_end = time.time()
    self.mv_time[2**self.param.level].append(mv_end-mv_start)

    return sol

#----------------------------------------------------------------------------#

  def compute_scattering_source(self,x) :
    """Compute the scattering source given a flux."""

    self.scattering_src = np.zeros((4*self.param.n_mom*self.param.n_cells))
    for cell in xrange(0,int(self.param.n_cells)) :
# Get i,j pair from a cell
      [i,j] = self.cell_mapping(cell)
      i_mat = self.param.mat_id[i,j]
      sca = self.param.sig_s[:,i_mat]
# Get location in the matrix
      ii = self.mapping(i,j)
# Block diagonal term
      for k in xrange(0,self.param.n_mom) :
        kk = k*4*self.param.n_cells+ii
        tmp = x[kk[0]:kk[3]+1]
        dot_product = np.dot(self.surface_sv,tmp)
        pos = 0
        for i_kk in xrange(int(kk[0]), int(kk[3]+1)) :
          self.scattering_src[i_kk] += self.scattering_src[i_kk]+sca[k]*\
              dot_product[pos]
          pos += 1
