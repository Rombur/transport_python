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
import quadrature
import finite_element
import p1sa

class transport_solver  :
  """Solve the transport equation in 2D for cartesian geometry"""

  def __init__(self,param,tol,max_it) :

    self.tol = tol
    self.max_iter = max_it
    self.param = param
# Create and build the quadrature
    self.quad = quadrature.quadrature(param)
    self.quad.build_quadrature()
# Create and build the finite element object
    self.fe = finite_element.finite_element(param)
    self.fe.build_2D_FE()
    self.fe.build_1D_FE()
# Compute the most normal direction
    self.most_normal()

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
    print 'L2-norm of the residual for iteration %i'%self.gmres_iteration +\
        ' : %f'%scipy.linalg.norm(residual)

#----------------------------------------------------------------------------#

  def solve(self) :
    """Solve the transport equation"""

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
      print 'Transport did not converge.'

    if self.param.is_precond==True :
      precond = p1sa.p1sa(self.param,self.quad,self.fe,self.tol/1e+2)
      delta = precond.solve(self.flux_moments)
      self.flux_moments += delta

# Solve the P1SA equation
    p1sa_src = self.compute_p1sa_src()
    p1sa_eq = p1sa.p1sa(self.param,self.quad,self.fe,self.tol)
    self.p1sa_flxm = p1sa_eq.solve(p1sa_src)
    
#----------------------------------------------------------------------------#

  def compute_p1sa_src(self) :
    """Compute the rhs when using the P1SA as solver."""

    x = np.zeros([3*4*self.param.n_cells])
    for i in xrange(0,self.param.n_y) :
      for j in xrange(0,self.param.n_x) :
        cell = j+i*self.param.n_x
        x[4*cell:4*(cell+1)] = self.param.src[self.param.src_id[j,i]]
    
    return x

#----------------------------------------------------------------------------#

  def mv(self,x) :
    """Perform the matrix-vector multiplication needed by GMRES."""

    y=x.copy()
    if self.param.is_precond==True :
      precond = p1sa.p1sa(self.param,self.quad,self.fe,self.tol/1e+2)
      delta = precond.solve(x.copy())
      y += delta

# Compute the scattering source
    self.compute_scattering_source(y)

# Do a transport sweep (no iteration on significant angular fluxes, we assume
# that no BC are reflective)
    flxm = self.sweep(False)
    sol = y-flxm
    
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

  def cell_mapping(self,cell) :
    """Get the i,j pair of a cell given a cell."""

    j = np.floor(cell/self.param.n_x)
    i = cell - j*self.param.n_x

    return i,j

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
