# Python code
# Author: Bruno Turcksin
# Date: 2011-04-01 10:15:51.682471

#----------------------------------------------------------------------------#
## Class transport_solver                                                   ##
#----------------------------------------------------------------------------#

"""Contain the solver of the transport equation. cell_mapping and mapping are
not class method so they can be accessed easily from outside the class. """

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
import parameters
import quadrature
import finite_element
import p1sa
import mip
import block_transport_solver

def cell_mapping(cell,n_x) :
  """Get the i,j pair of a cell given a cell."""

  j = np.floor(cell/n_x)
  i = cell - j*n_x

  return i,j

#----------------------------------------------------------------------------#

def mapping(i,j,n_x) :
  """Compute the index in the 'local' matrix."""

  cell = int(i+j*n_x)
  index = range(4*cell,4*(cell+1))

  return index

#----------------------------------------------------------------------------#

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
# Create and build the finite element object
    self.fe = finite_element.finite_element(param)
    self.fe.build_2D_FE()
    self.fe.build_1D_FE()
# Compute the upwind edge matrices
    self.upwind_edge()
# Compute the most normal direction
    if np.fmod(self.param.sn,2)==0 :
       self.most_normal()
# Save the values to be printed in a file 
    self.output_file = output_file
# List of time for each MV
    self.mv_time = mv_time
# List of time for each MIP
    self.mip_time = mip_time

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
      coarse_solver = transport_solver(coarse_param,self.tol,self.max_iter,
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
        flux_moments_old = self.initial_guess()
        for i in xrange(0,self.max_iter) :
          if self.param.block_solver==True :
            if i==0 :
              self.all_psi = []
              for idir in xrange(0,self.quad.n_dir) :
                self.all_psi.append(np.zeros((4*self.param.n_cells)))
            block_solver = block_transport_solver.block_transport_solver(\
                self.param,self.fe,self.quad,flux_moments_old,self.all_psi,\
                self.x_down,self.x_up,self.y_down,self.y_up,self.most_n,\
                self.tol,self.max_iter)
            self.flux_moments = block_solver.solve()
            flux_moments_old = self.flux_moments.copy()

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
#    p1sa_src = self.compute_precond_src('P1SA')
#    p1sa_eq = p1sa.p1sa(self.param,self.fe,self.tol,self.output_file)
#    self.p1sa_flxm = p1sa_eq.solve(p1sa_src)
    self.p1sa_flxm = self.flux_moments.copy()

# Solve the MIP equation
#    mip_src = self.compute_precond_src('MIP')
#    self.param.projection = 'scalar'
#    mip_eq = mip.mip(self.param,self.fe,self.tol,self.output_file)
#    self.mip_flxm = mip_eq.solve(mip_src)
    self.mip_flxm = self.flux_moments.copy()
    
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
        coarse_solver = transport_solver(coarse_param,self.tol,self.max_iter,
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
        coarse_solver = transport_solver(coarse_param,self.tol,self.max_iter,
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
      [i,j] = cell_mapping(cell,self.param.n_x)
      i_mat = self.param.mat_id[i,j]
      sca = self.param.sig_s[:,i_mat]
# Get location in the matrix
      ii = mapping(i,j,self.param.n_x)
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

    self.x_down= np.zeros((2,4,4))
    self.x_up = np.zeros((2,4,4))
    self.y_down = np.zeros((2,4,4))
    self.y_up = np.zeros((2,4,4))

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
          self.x_down[k,x_down_i[k,i],x_down_j[k,j]] = (-1)**k *\
              self.fe.vertical_edge_mass_matrix[i,j]
          self.x_up[k,x_up_i[k,i],x_up_j[k,j]] = (-1)**(k+1) *\
              self.fe.vertical_edge_mass_matrix[i,j]
          self.y_down[k,y_down_i[k,i],y_down_j[k,j]] = (-1)**k *\
              self.fe.horizontal_edge_mass_matrix[i,j]
          self.y_up[k,y_up_i[k,i],y_up_j[k,j]] = (-1)**(k+1) *\
              self.fe.horizontal_edge_mass_matrix[i,j]

#----------------------------------------------------------------------------#

  def sweep(self,gmres_rhs) :
    """Do the transport sweep on all the directions."""
                                                    
    tmp = int(4*self.param.n_cells)
    self.scattering_src = self.scattering_src.reshape(self.param.n_mom,tmp)

    flux_moments = np.zeros((4*self.param.n_mom*self.param.n_cells))
    if self.param.block_solver==True :
      self.all_psi = []
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
      gradient = omega_x*(-self.fe.x_grad_matrix+self.x_down[sx,:,:])+omega_y*(
          -self.fe.y_grad_matrix+self.y_down[sy,:,:])
      
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
          ii = mapping(i,j,self.param.n_x)

# Add scattering source contribution
          scat_src = np.dot(self.quad.M[idir,:],self.scattering_src[:,ii])
          rhs += scat_src

# Block diagonal term
          L = gradient+sig_t*self.fe.mass_matrix

# Upwind term in x
          if i>0 and sx==0 :
            jj = mapping(i-1,j,self.param.n_x)
            rhs -= omega_x*np.dot(self.x_up[sx,:,:],psi[jj])
          else :
            if i==0 and idir in self.most_n['left'] and gmres_rhs==True :
              rhs -= omega_x*np.dot(self.x_up[sx,:,:],self.param.inc_left[j]*\
                  np.ones((4)))
          if i<self.param.n_x-1 and sx==1 :
            jj = mapping(i+1,j,self.param.n_x)
            rhs -= omega_x*np.dot(self.x_up[sx,:,:],psi[jj])
          else :
            if i==self.param.n_x-1 and idir in self.most_n['right'] and\
                gmres_rhs==True :
              rhs -= omega_x*np.dot(self.x_up[sx,:,:],self.param.inc_right[j]*\
                  np.ones((4)))

# Upwind term in y
          if j>0 and sy==0 :
            jj = mapping(i,j-1,self.param.n_x)
            rhs -= omega_y*np.dot(self.y_up[sy,:,:],psi[jj])
          else :
            if j==0 and idir in self.most_n['bottom'] and gmres_rhs==True :
              rhs -= omega_y*np.dot(self.y_up[sy,:,:],self.param.inc_bottom[i]*\
                  np.ones((4)))
          if j<self.param.n_y-1 and sy==1 :
            jj = mapping(i,j+1,self.param.n_x)
            rhs -= omega_y*np.dot(self.y_up[sy,:,:],psi[jj])
          else :
            if j==self.param.n_y-1 and idir in self.most_n['top'] and\
                gmres_rhs==True :
              rhs -= omega_y*np.dot(self.y_up[sy,:,:],self.param.inc_top[i]*\
                  np.ones((4)))

          psi[ii] = scipy.linalg.solve(L,rhs,sym_pos=False,lower=False,
            overwrite_a=True,overwrite_b=True)

# update scalar flux
      for i in xrange(0,self.param.n_mom) :
        i_begin = i*4*self.param.n_cells
        i_end = (i+1)*4*self.param.n_cells
        flux_moments[i_begin:i_end] += self.quad.D[i,idir]*psi[:]
        
      if self.param.block_solver==True :
        self.all_psi.append(psi)
    
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

  def initial_guess(self) :
    """Initial guess of the flux moment."""

    flux = np.zeros(4*self.param.n_cells)
    for cell in xrange(0,self.param.n_cells) :
# Compute the row of the cell 
      j = np.floor(cell/self.param.n_x)
      if np.mod(j,2)==0 :
        flux[4*cell] = 1.
        flux[4*cell+2] = 1.
        flux[4*cell+1] = -1.
        flux[4*cell+3] = -1.
      else :
        flux[4*cell] = -1.
        flux[4*cell+2] = -1.
        flux[4*cell+1] = 1.
        flux[4*cell+3] = 1.

    return flux
