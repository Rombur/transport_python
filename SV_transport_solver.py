# Python code
# Author: Bruno Turcksin
# Date: 2011-07-04 15:32:11.045280

#----------------------------------------------------------------------------#
## Class SV_transport_solver                                                ##
#----------------------------------------------------------------------------#

"""Contain the spectral volumes solver of the transport equation"""

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time 
import transport_solver as solver
import spectral_volume

class SV_transport_solver(solver.transport_solver) :
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
      self.scattering_src = np.zeros((self.param.n_mom,4*self.param.n_cells))
      self.gmres_b = self.sweep(gmres_rhs)

# GMRES solver 
      self.gmres_iteration = 0 
      size = self.gmres_b.shape[0]
# NEED TO COMPUTE THE RHS
      A = scipy.sparse.linalg.LinearOperator((size,size),matvec=self.mv,
          rmatvec=None,dtype=float)
      self.flux_moments,flag = scipy.sparse.linalg.gmres(A,self.gmres_b,
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

    self.p1sa_flxm = self.flux_moments.copy()
    self.mip_flxm = self.flux_moments.copy()

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
        dot_product = np.dot(self.sv.surface_sv,tmp)
        pos = 0
        for i_kk in xrange(int(kk[0]), int(kk[3]+1)) :
          self.scattering_src[i_kk] += self.scattering_src[i_kk]+sca[k]*\
              dot_product[pos]
          pos += 1

#----------------------------------------------------------------------------#

  def upwind_edge(self) :
    """Compute the edge for the upwind."""

    x_down = np.zeros((2,4,4))
    x_up = np.zeros((2,4,4))
    y_down = np.zeros((2,4,4))
    y_up = np.zeros((2,4,4))

    x_down[0,0,:] = self.sv.right_edge_cv_0[:]
    x_down[0,1,:] = self.sv.right_edge_cv_1[:]-self.sv.left_edge_cv_1[:]
    x_down[0,2,:] = self.sv.right_edge_cv_2[:]-self.sv.left_edge_cv_2[:]
    x_down[0,3,:] = self.sv.right_edge_cv_3[:]

    x_down[1,0,:] = -self.sv.left_edge_cv_0[:]+self.sv.right_edge_cv_0[:]
    x_down[1,1,:] = -self.sv.left_edge_cv_1[:]
    x_down[1,2,:] = -self.sv.left_edge_cv_2[:]
    x_down[1,3,:] = -self.sv.left_edge_cv_3[:]+self.sv.right_edge_cv_3[:]

    y_down[0,0,:] = self.sv.top_edge_cv_0[:]
    y_down[0,1,:] = self.sv.top_edge_cv_1[:]
    y_down[0,2,:] = self.sv.top_edge_cv_2[:]-self.sv.bottom_edge_cv_2[:]
    y_down[0,3,:] = self.sv.top_edge_cv_3[:]-self.sv.bottom_edge_cv_3[:]

    y_down[1,0,:] = -self.sv.bottom_edge_cv_0[:]+self.sv.top_edge_cv_0[:]
    y_down[1,1,:] = -self.sv.bottom_edge_cv_1[:]+self.sv.top_edge_cv_1[:]
    y_down[1,2,:] = -self.sv.bottom_edge_cv_2[:]
    y_down[1,3,:] = -self.sv.bottom_edge_cv_3[:]

    x_up[0,0,:] = -self.sv.right_edge_cv_1[:]
    x_up[0,3,:] = -self.sv.right_edge_cv_2[:]

    x_up[1,1,:] = self.sv.left_edge_cv_0[:]
    x_up[1,2,:] = self.sv.left_edge_cv_3[:]

    y_up[0,0,:] = -self.sv.top_edge_cv_3[:]
    y_up[0,1,:] = -self.sv.top_edge_cv_2[:]

    y_up[1,2,:] = self.sv.bottom_edge_cv_1[:]
    y_up[1,3,:] = self.sv.bottom_edge_cv_0[:]

    return x_down,x_up,y_down,y_up

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

# Upwind/downwinf indices
      if omega_x>0.0 :
        sx = 0
        x_begin = 0
        x_end = int(self.param.n_x)
        x_incr = 1
      else :
        sx = 1
        x_begin = int(self.param.n_x-1)
        x_end = -1
        x_incr = -1
      if omega_y>0.0 :
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
      gradient = omega_x*x_down[sx,:,:]+omega_y*y_down[sy,:,:]

      for i in xrange(x_begin,x_end,x_incr) :
        for j in xrange(y_begin,y_end,y_incr) :
          i_mat = self.param.mat_id[i,j]
          sig_t = self.param.sig_t[i_mat]

# Volumetric term of the rhs
          i_src = self.param.src_id[i,j]
          rhs = np.zeros((4))
          if gmres_rhs==True :
            rhs = self.param.src[i_src]*self.sv.surface_sv

# Get location in the matrix
          ii = self.mapping(i,j)

# Add scattering source contribution
          scat_src = np.dot(self.quad.M[idir,:],self.scattering_src[:,ii])
          rhs += scat_src

# Block diagonal term
          L = gradient+sig_t*self.sv.surface_sv

# Upwind term in x 
          if i>0 and sx==0 :
            jj = self.mapping(i-1,j)
            rhs -= omega_x*np.dot(x_up[sx,:,:],psi[jj])
          elif i==0 and idir in self.most_n['left'] and gmres_rhs==True :
              rhs -= omega_x*np.dot(x_up[sx,:,:],self.param.inc_left[j]*\
                  np.ones((4)))
          if i<self.param.n_x-1 and sx==1 :
            jj = self.mapping(i+1,j)
            rhs -= omega_x*np.dot(x_up[sx,:,:],psi[jj])
          elif i==self.param.n_x-1 and idir in self.most_n['right'] and\
              gmres_rhs==True :
                rhs -= omega_x*np.dot(x_up[sx,:,:],self.param.inc_right[j]*\
                    np.ones((4)))

# Upwind term in y 
          if j>0 and sy==0 :
            jj = self.mapping(i,j-1)
            rhs -= omega_y*np.dot(y_up[sy,:,:],psi[jj])
          elif j==0 and idir in self.most_n['bottom'] and gmres_rhs==True :
            rhs -= omega_y*np.dot(y_up[sy,:,:],self.param.inc_bottom[i]*\
                np.ones((4)))
          if j<self.param.n_y-1 and sy==1 :
            jj = self.mapping(i,j+1)
            rhs -= omega_y*np.dot(y_up[sy,:,:],psi[jj])
          elif j==self.param.n_y-1 and idir in self.most_n['top'] and\
              gmres_rhs==True :
                rhs -= omega_y*np.dot(y_up[sy,:,:],self.param.inc_top[i]*\
                    np.ones((4)))

          psi[ii] = scipy.linalg.solve(L,rhs,sym_pos=False,lower=False,
            overwrite_a=True,overwrite_b=True)

# Update scalar flux
      for i in xrange(0,self.param.n_mom) :
        i_begin = i*4*self.param.n_cells
        i_end = (i+1)*4*self.param.n_cells
        flux_moments[i_begin:i_end] += self.quad.D[i,idir]*psi[:]

    return flux_moments
