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

class transport_solver  :
  """Solve the transport equation in 2D for cartesian geometry"""

  def __init__(self,param,tol,max_it) :

    self.tol = tol
    self.max_it = max_it
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
    res = scipy.linalg.normv(residual) 
    print 'Iteration : %i'%self.gmres_iteration
    print 'L2 of the residual : %d'%scipy.linalg.norm(residual)

#----------------------------------------------------------------------------#

  def solve(self) :
    """Solve the transport equation"""

# Compute the uncollided flux moment (D*inv(L)*q) = rhs of gmres
    gmres_rhs = True
    self.scattering_src = np.zeros((self.param.n_mom,4*self.param.n_cells))
    self.gmres_b = self.sweep(gmres_rhs)

# GMRES solver 
    self.gmres_iteration = 0
    print '?'
    self.flux_moment = scipy.sparse.linalg.gmres(A,b,x0=None,tol=self.tol,
        restrt=None,maxiter=self.max_iter,M=None,
        callback=self.cout_gmres_iteration)

#----------------------------------------------------------------------------#

  def upwind_edge(self) :
    """Compute the edge for the upwind."""

    x_down= np.zeros((2,4,4))
    x_up = np.zeros((2,4,4))
    y_down = np.zeros((2,4,4))
    y_up = np.zeros((2,4,4))

    x_down_i = [[2,3],[0,1]]
    x_up_i = [[0,1],[2,3]]
    x_down_j = [[2,3],[0,1]]
    x_up_j = [[2,3],[0,1]]

    y_down_i = [[1,3],[0,2]]
    y_up_i = [[0,2],[1,3]]
    y_down_j = [[1,3],[0,2]]
    y_up_j = [[1,3],[0,2]]

    for k in xrange(0,2) :
      for i in xrange(0,2) :
        for j in xrange(0,2) :
          x_down[k,x_down_i[i],x_down_j[j]] = (-1)**k *\
              self.fe.vertical_edge_mass_matrix[i,j]
          x_up[k,x_up_i[i],x_up_j[j]] = (-1)**(k+1) *\
              self.fe.vertical_edge_mass_matrix[i,j]
          y_down[k,y_down_i[i],y_down_j[j]] = (-1)**k *\
              self.fe.horizontal_edge_mass_matrix[i,j]
          y_up[k,y_up_i[i],y_up_j[j]] = (-1)**(k+1) *\
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
                                                    
    [x_down,x_up,y_down,y_up] = self.upwind_edge()
    
    flux_moments = np.zeros((4*self.param.n_mom*self.param.n_cells))
    for idir in xrange(0,self.quad.n_dir) :
      psi=np.zeros((4*self.param.n_cells))

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

# Volumetix term of the rhs
          i_src = self.param.src_id[i,j]
          rhs = np.zeros((4))
          if gmres_rhs == True :
            rhs = self.param.src[i_src]*self.fe.width_cell[0]*\
                self.fe.width_cell[1]*np.ones((4))/4.
# Put the source to 0 because it is only need to build the rhs for GMRES

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
        flux_moments[i_begin:i_end] += self.quad.M[i,idir]*psi[:]
    
    return flux_moments
