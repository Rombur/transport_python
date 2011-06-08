# Python code
# Author: Bruno Turcksin
# Date: 2011-06-08 12:28:45.930931

#----------------------------------------------------------------------------#
## Class FE_transport_solver                                                ##
#----------------------------------------------------------------------------#

"""Contain the finite elements solver of the transport equation"""

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
import transport_solver as solver
import parameters
import finite_element
import p1sa
import mip

class FE_transport_solver(solver.transport_solver)  :
  """Solve the transport equation discretized with finite elements."""

  def __init__(self,param,tol,max_it,output_file,mv_time=None,
      mip_time=None) :

    super(FE_transport_solver,self).__init__(param,tol,max_it,output_file)

# Create and build the finite element object
    self.fe = finite_element.finite_element(param)
    self.fe.build_2D_FE()
    self.fe.build_1D_FE()
# List of time for each MV
    self.mv_time = mv_time
# List of time for each MIP
    self.mip_time = mip_time

#----------------------------------------------------------------------------#

  def solve(self) :
    """Solve the transport equation"""

    start = time.time()
    self.mv_time = [[] for i in xrange(0,self.param.sn+1)]
    self.mip_time = []
    if self.param.multigrid==True :
# Compute the uncollided flux moment (D*inv(L)*q) = rhs of gmres
      gmres_rhs = True
      self.scattering_src = np.zeros((self.param.n_mom,4*self.param.n_cells))
      self.gmres_b = self.sweep(gmres_rhs)

# GMRES solver 
      self.gmres_iteration = 0                
      size = self.gmres_b.shape[0]
      A = scipy.sparse.linalg.LinearOperator((size,size),matvec=self.mv,
          rmatvec=None,dtype=float)
      self.flux_moments,flag = scipy.sparse.linalg.gmres(A,self.gmres_b,x0=None,
          tol=self.tol,restrt=20,maxiter=self.max_iter,M=None,
          callback=self.count_gmres_iterations)

      if flag != 0 :
        self.print_message('Transport did not converge.')

      y = self.flux_moments.copy()
      coarse_param = parameters.parameters(self.param.galerkin,
          self.param.fokker_planck,self.param.TC,self.param.optimal,
          self.param.preconditioner,self.param.multigrid,self.param.L_max/2,
          self.param.sn/2,level=1,max_level=self.param.max_level)
      coarse_solver = FE_transport_solver(coarse_param,self.tol,self.max_iter,
          self.output_file,mv_time=self.mv_time,mip_time=self.mip_time)
      solution = coarse_solver.mv(y)
      self.flux_moments += self.project_vector(solution)
    else :
      if self.param.gmres==True :
# Compute the uncollided flux moment (D*inv(L)*q) = rhs of gmres
        gmres_rhs = True
        self.scattering_src = np.zeros((self.param.n_mom,4*self.param.n_cells))
        self.gmres_b = self.sweep(gmres_rhs)

# GMRES solver 
        self.gmres_iteration = 0                
        size = self.gmres_b.shape[0]
        A = scipy.sparse.linalg.LinearOperator((size,size),matvec=self.mv,
            rmatvec=None,dtype=float)
        self.flux_moments,flag = scipy.sparse.linalg.gmres(A,self.gmres_b,x0=None,
            tol=self.tol,restrt=20,maxiter=self.max_iter,M=None,
            callback=self.count_gmres_iterations)

        if flag != 0 :
          self.print_message('Transport did not converge.')

        if self.param.preconditioner=='P1SA' :
          precond = p1sa.p1sa(self.param,self.fe,self.tol/1e+2,self.output_file)
          delta = precond.solve(self.flux_moments)
          self.flux_moments += delta
        elif self.param.preconditioner=='MIP' :
          precond = mip.mip(self.param,self.fe,self.tol/1e+2,self.output_file)
          mip_start = time.time()
          delta = precond.solve(self.flux_moments)
          mip_end = time.time()
          self.mip_time.append(mip_end-mip_start)
          self.flux_moments += delta
      else :
        rhs = True
        self.flux_moments = np.zeros((4*self.param.n_cells*self.param.n_mom))
        flux_moments_old = np.zeros((4*self.param.n_cells*self.param.n_mom))
        for i in xrange(0,self.max_iter) :
          self.compute_scattering_source(flux_moments_old)
          self.flux_moments = self.sweep(rhs)

          if self.param.preconditioner=='P1SA' :
            precond = p1sa.p1sa(self.param,self.fe,self.tol/1e+2,self.output_file)
            delta = precond.solve(self.flux_moments-flux_moments_old)
            self.flux_moments += delta

          elif self.param.preconditioner=='MIP' :
            precond = mip.mip(self.param,self.fe,self.tol/1e+2,self.output_file)
            mip_start = time.time()
            delta = precond.solve(self.flux_moments-flux_moments_old)
            mip_end = time.time()
            self.mip_time.append(mip_end-mip_start)
            self.flux_moments += delta

          conv = scipy.linalg.norm(self.flux_moments-flux_moments_old)/\
              scipy.linalg.norm(self.flux_moments)

          self.print_message('L2-norm of the residual for iteration %i'\
              %i+' : %f'%conv)
          if conv<self.tol :
            break

          flux_moments_old = self.flux_moments.copy()

    end = time.time()

    if self.param.verbose >= 1:
      mv_time2 = [x for x in self.mv_time if x!=[]]
      for i_m in xrange(0,len(mv_time2)) :
        self.print_message('MV time : '+str(mv_time2[i_m]))
        self.print_message('Sum mv time : '+str(sum(mv_time2[i_m])))
      if self.param.preconditioner=='MIP' :
        self.print_message('MIP time : '+str(self.mip_time))
        self.print_message('Sum MIP time : '+str(sum(self.mip_time)))

    self.print_message('Elapsed time to solve the problem : %f'%(end-start))

# Solve the P1SA equation
    p1sa_src = self.compute_precond_src('P1SA')
    p1sa_eq = p1sa.p1sa(self.param,self.fe,self.tol,self.output_file)
    self.p1sa_flxm = p1sa_eq.solve(p1sa_src)

# Solve the MIP equation
    mip_src = self.compute_precond_src('MIP')
    self.param.projection = 'scalar'
    mip_eq = mip.mip(self.param,self.fe,self.tol,self.output_file)
    self.mip_flxm = mip_eq.solve(mip_src)

#----------------------------------------------------------------------------#

  def compute_precond_src(self,precond) :
    """Compute the rhs when using the P1SA or the MIP as solver."""

    if precond=='P1SA' :
      x = np.zeros([3*4*self.param.n_cells])
    else :
      x = np.zeros([4*self.param.n_cells])
    for i in xrange(0,self.param.n_y) :
      for j in xrange(0,self.param.n_x) :
        cell = j+i*self.param.n_x
        x[4*cell:4*(cell+1)] = self.param.src[self.param.src_id[j,i]]
    
    return x

#----------------------------------------------------------------------------#

  def mv(self,x) :
    """Perform the matrix-vector multiplication needed by GMRES."""

    mv_start = time.time()
    y=x.copy()
    if self.param.multigrid==True :
      if self.param.level==0 :
        coarse_param = parameters.parameters(self.param.galerkin,
            self.param.fokker_planck,self.param.TC,self.param.optimal,
            self.param.preconditioner,self.param.multigrid,self.param.L_max/2,
            self.param.sn/2,level=1,max_level=self.param.max_level)
        coarse_solver = FE_transport_solver(coarse_param,self.tol,self.max_iter,
            self.output_file,mv_time=self.mv_time,mip_time=self.mip_time)
        solution = coarse_solver.mv(y)
        y += self.project_vector(solution)

# Compute the scattering source
        self.compute_scattering_source(y)

# Do a transport sweep (no iteration on significant angular fluxes, we assume
# that no BC are reflective)
        flxm = self.sweep(False)
        sol = y-flxm
      elif self.param.level>self.param.max_level :
        if self.param.preconditioner=='MIP' :
          precond = mip.mip(self.param,self.fe,self.tol/1e+2,self.output_file)
          mip_start = time.time()
          delta = precond.solve(y)
          mip_end = time.time()
          self.mip_time.append(mip_end-mip_start)
          sol = delta
        elif self.param.preconditioner=='P1SA' :
          precond = p1sa.p1sa(self.param,self.fe,self.tol/1e+2,
              self.output_file)
          delta = precond.solve(y)
          sol = delta
        else :
          sol = y
      else :
        z = self.restrict_vector(y)
        if self.param.sn!=6 :
          new_L_max = self.param.L_max/2
          new_sn = self.param.sn/2
          new_level = self.param.level+1
        else :
          new_L_max = 4
          new_sn = 4
          new_level = self.param.level+0.5849625007211562

        coarse_param = parameters.parameters(self.param.galerkin,
            self.param.fokker_planck,self.param.TC,self.param.optimal,
            self.param.preconditioner,self.param.multigrid,new_L_max,
            new_sn,level=new_level,max_level=self.param.max_level)
        coarse_solver = FE_transport_solver(coarse_param,self.tol,self.max_iter,
            self.output_file,mv_time=self.mv_time,mip_time=self.mip_time)
        solution = coarse_solver.mv(z)
        solution_proj = self.project_vector(solution)
        z += solution_proj

# Compute the scattering source
        self.compute_scattering_source(z)

# Do a transport sweep (no iteration on significant angular fluxes, we assume
# that no BC are reflective)
        flxm = self.sweep(False)
        sol = flxm
    else :
      if self.param.preconditioner=='P1SA' :
        precond = p1sa.p1sa(self.param,self.fe,self.tol/1e+2,self.output_file)
        delta = precond.solve(x.copy())
        y += delta
      elif self.param.preconditioner=='MIP' :
        precond = mip.mip(self.param,self.fe,self.tol/1e+2,self.output_file)
        mip_start = time.time()
        delta = precond.solve(x.copy())
        mip_end = time.time()
        self.mip_time.append(mip_end-mip_start)
        y += delta

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
    """Compute the scattering given a flux."""

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
        kk = k*4*self.param.n_cells + ii
        tmp = x[kk[0]:kk[3]+1]
        dot_product = np.dot(self.fe.mass_matrix,tmp)
        pos = 0
        for i_kk in xrange(int(kk[0]),int(kk[3]+1)) :
          self.scattering_src[i_kk] += self.scattering_src[i_kk] + sca[k]*\
              dot_product[pos]
          pos += 1

#----------------------------------------------------------------------------#

  def upwind_edge(self) :
    """Compute the edge for the upwind."""

    x_down= np.zeros((2,4,4))
    x_up = np.zeros((2,4,4))
    y_down = np.zeros((2,4,4))
    y_up = np.zeros((2,4,4))

    x_down_i = np.array([[2,3],[0,1]])
    x_up_i = np.array([[0,1],[2,3]])
    x_down_j = np.array([[2,3],[0,1]])
    x_up_j = np.array([[2,3],[0,1]])

    y_down_i = np.array([[1,3],[0,2]])
    y_up_i = np.array([[0,2],[1,3]])
    y_down_j = np.array([[1,3],[0,2]])
    y_up_j = np.array([[1,3],[0,2]])

    for k in xrange(0,2) :
      for i in xrange(0,2) :
        for j in xrange(0,2) :
          x_down[k,x_down_i[k,i],x_down_j[k,j]] = (-1)**k *\
              self.fe.vertical_edge_mass_matrix[i,j]
          x_up[k,x_up_i[k,i],x_up_j[k,j]] = (-1)**(k+1) *\
              self.fe.vertical_edge_mass_matrix[i,j]
          y_down[k,y_down_i[k,i],y_down_j[k,j]] = (-1)**k *\
              self.fe.horizontal_edge_mass_matrix[i,j]
          y_up[k,y_up_i[k,i],y_up_j[k,j]] = (-1)**(k+1) *\
              self.fe.horizontal_edge_mass_matrix[i,j]

    return x_down,x_up,y_down,y_up
