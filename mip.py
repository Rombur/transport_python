# Python code
# Author: Bruno Turcksin
# Date: 2011-04-18 18:34:10.749902

#----------------------------------------------------------------------------#
## Class mip                                                                ##
#----------------------------------------------------------------------------#

"""Solve the MIP equation"""

import numpy as np
import scipy.sparse
import scipy.linalg
import scipy.sparse.extract
import scipy.sparse.linalg
import pyamg
import cg
import synthetic_acceleration as sa

class mip(sa.synthetic_acceleration) :
  """Preconditioner for the transport using the MIP equation."""

  def __init__(self,parameters,fe,tol,output_file) :

    super(mip,self).__init__(parameters,fe,tol,output_file)

#----------------------------------------------------------------------------#

  def count_cg_iterations(self,residual) :
    """Callback function called at the end of each CG iteration. Count the
    number of iterations and compute the L2 norm of the current residual."""

    self.cg_iteration += 1
    res = scipy.linalg.norm(residual)
    if self.param.verbose==2 :
      self.print_message(' L2-norm of the residual for iteration %i'\
          %self.cg_iteration + ' : %f'%scipy.linalg.norm(residual))

#----------------------------------------------------------------------------#

  def solve(self,x) :
    """Solve the MIP equation where the right-hand-side is a Krylov vector."""

# Restrict the Krylov vector to build the rhs
    self.size = int(4*self.param.n_cells)
    self.mip_b = np.zeros([self.size])
    self.krylov_vector = np.zeros([self.size])
    self.krylov_vector[:] = x[0:self.size]

# Compute the right-hand-side
    self.compute_rhs()

# CG solver
    self.cg_iteration = 0
    if self.param.matrix_free==True : 
      A = scipy.sparse.linalg.LinearOperator((self.size,self.size),matvec=self.mv,
        rmatvec=None,dtype=float)
      flux,flag = scipy.sparse.linalg.cg(A,self.mip_b,tol=self.tol,
          callback=self.count_cg_iterations)
      if flag!=0 :
        self.print_message('MIP did not converge')
    else :
      A = self.build_lhs()
      A = A.tocsr()
      if self.param.pyamg==False :  
        if self.param.my_cg==True :
          rows,columns,values = scipy.sparse.extract.find(A)
          flux = np.zeros([self.size])
          iteration = np.array([0],dtype='int32')
          cg.pcg(values,rows,columns,self.mip_b,flux,self.tol,iteration)
          if self.param.verbose==2 :
            self.print_message(' Converged after %i'%iteration+' iterations')
        else :
          P = self.compute_ssor(A)
          flux,flag = scipy.sparse.linalg.cg(A,self.mip_b,tol=self.tol,
              M=P,callback=self.count_cg_iterations)
          if flag!=0 :
            self.print_message('MIP did not converge')
      else :  
        resvec = []
        Agg = pyamg.aggregation.standard_aggregation(A)
        B = np.ones([A.shape[0],1])
        ml = pyamg.smoothed_aggregation_solver(A,B,max_coarse=10)
        if self.param.accel==True :
          flux = ml.solve(b=self.mip_b,tol=self.tol,accel='cg',residuals=resvec)
        else :
          flux = ml.solve(b=self.mip_b,tol=self.tol,residuals=resvec)
        if self.param.verbose==2 :
          rho = pyamg.util.linalg.approximate_spectral_radius(A,symmetric=True)
          self.print_message( 'The approximate spectral radius is %f'%rho)
          self.print_message(ml)
          self.print_message(resvec)

# Project the MIP solution on the whole space
    krylov_space_size = x.shape[0]
    output = np.zeros([krylov_space_size])
    output[0:self.size] = flux[:]

    return output

#----------------------------------------------------------------------------#

  def compute_rhs(self) :
    """Compute the rhs of the MIP equation."""

    for cell in xrange(0,self.param.n_cells) :
      for i in xrange(0,self.fe.n_dofs_per_cell) :
        pos_i = self.index(i,cell)
        for j in xrange(0,self.fe.n_dofs_per_cell) :
          pos_j = self.index(j,cell)
          self.mip_b[pos_i] += self.fe.mass_matrix[i,j]*\
              self.krylov_vector[pos_j]

#----------------------------------------------------------------------------#

  def compute_ssor(self,A) :
    """Compute the SSOR preconditioner used with CG when the left-hand-side is
    build."""

    # Because of some limitations of lil_matrix, the following method is used
    # to build the SSOR preconditioner : extract the lower triangular part of A,
    # solve L x_i = e_i, put x_i at the ith row of inv_L = compute the invert
    # of the upper triangular part of A, then compute
    # inv_U*D*inv_U.transpose()
    D = scipy.sparse.lil_matrix((self.size,self.size))
    L = scipy.sparse.extract.tril(A)
    inv_U = scipy.sparse.lil_matrix((self.size,self.size))

    for i in xrange(0,self.size) :
      b = scipy.sparse.lil_matrix((self.size,1))
      b[i,0] = 1.
      b = b.tocsr()
      x = scipy.sparse.linalg.spsolve(L,b)
      inv_U[i,:] = x[:]
      D[i,i] = A[i,i]
    
    D = D.tocsr()
    inv_U = inv_U.tocsr()

    M = D*inv_U.transpose()
    M = inv_U*M
    M = M.tocsr()
    
    return M

#----------------------------------------------------------------------------#

  def compute_diffusion_coefficient(self,dof,edge_offset) :
    """Compute the diffusion coeffricients of D^+ and D^-."""

    dofs_per_cell = 4
    cell = np.floor(dof/dofs_per_cell)
    i,j = self.cell_mapping(cell) 
    i_mat = self.param.mat_id[i,j]
    sig_a = self.param.sig_t[i_mat]-self.param.sig_s[0,i_mat] 
# Compute the diffusion coefficient D
    if self.param.sig_s.shape[0]>1 :
      sig_tr = self.param.sig_t[i_mat]-self.param.sig_s[1,i_mat]
    else :
      sig_tr = self.param.sig_t[i_mat]
    D_minus = 1./(3.*sig_tr)

    if edge_offset!=0 :
      cell = np.floor((dof+edge_offset)/dofs_per_cell)
      i,j = self.cell_mapping(cell) 
      i_mat = self.param.mat_id[i,j]
      sig_a = self.param.sig_t[i_mat]-self.param.sig_s[0,i_mat] 
# Compute the diffusion coefficient D
      if self.param.sig_s.shape[0]>1 :
        sig_tr = self.param.sig_t[i_mat]-self.param.sig_s[1,i_mat]
      else :
        sig_tr = self.param.sig_t[i_mat]
      D_plus = 1./(3.*sig_tr)
    else :
      D_plus = 0.

    return D_minus,D_plus

#----------------------------------------------------------------------------#

  def compute_penalty_coefficient(self,dof,edge_offset,D_minus,D_plus,h_minus,
      h_plus,interior) :
    """Compute the penalty coeffiecient used by MIP"""

    C = 2.
# Only use first order polynomial -> c(p^+) = c(p^-)
    p = 1.
    c_p = C*p*(p+1)
    if interior==True :
      k = c_p/2.*(D_plus/h_plus+D_minus/h_minus)
    else :
      k = c_p*D_minus/h_minus

    k_mip = np.max(k,0.25)

    return k_mip

#----------------------------------------------------------------------------#

  def mv(self,x_krylov) :
    """Perform the matrix-vector multiplication needed by CG. Only
    homogeneous Dirichlet conditions are implemented."""

    x = np.zeros([self.size])

    for cell in xrange(0,self.param.n_cells) :
      i,j = self.cell_mapping(cell) 
      i_mat = self.param.mat_id[i,j]
      sig_a = self.param.sig_t[i_mat]-self.param.sig_s[0,i_mat] 
# Compute the diffusion coefficient D
      if self.param.sig_s.shape[0]>1 :
        sig_tr = self.param.sig_t[i_mat]-self.param.sig_s[1,i_mat]
      else :
        sig_tr = self.param.sig_t[i_mat]
      D = 1./(3.*sig_tr)

      for i in xrange(0,self.fe.n_dofs_per_cell) :
        pos_i = self.index(i,cell)
        for j in xrange(0,self.fe.n_dofs_per_cell) :
          pos_j = self.index(j,cell)

          x[pos_i] += sig_a*self.fe.mass_matrix[i,j]*x_krylov[pos_j]
          x[pos_i] += D*self.fe.stiffness_matrix[i,j]*x_krylov[pos_j]

# The edges are numbered as follow : first the vertical ones (left to right
# and then the bottom to top) then the horizontal ones (bottom to top and left
# to right).
#     4
#   -----
#   |   |
# 1 |   | 2
#   |   | 
#   -----
#     3

    n_edges = self.param.n_x*(self.param.n_y+1)+self.param.n_y*\
        (self.param.n_x+1)
    for edge in xrange(0,n_edges) :
      inside = self.interior(edge)
      if inside == True :
        is_vertical = self.compute_vertical(edge,inside) 
        if is_vertical==True :
          edge_offset = self.compute_edge_offset('right')
          edge_mass_matrix = self.fe.vertical_edge_mass_matrix
          edge_deln_matrix = self.fe.edge_deln_matrix['right']
          outside_edge_deln_matrix = -self.fe.edge_deln_matrix['left']
          in_across_edge_deln_matrix = self.fe.across_edge_deln_matrix['right']
          out_across_edge_deln_matrix =\
              -self.fe.across_edge_deln_matrix['left']
          h_minus = self.fe.width_cell[0]
          h_plus = self.fe.width_cell[0]
          next_cell_offset = 1
        else :
          edge_offset = self.compute_edge_offset('top')
          edge_mass_matrix = self.fe.horizontal_edge_mass_matrix
          edge_deln_matrix = self.fe.edge_deln_matrix['top']
          outside_edge_deln_matrix = -self.fe.edge_deln_matrix['bottom']
          in_across_edge_deln_matrix = self.fe.across_edge_deln_matrix['top']
          out_across_edge_deln_matrix =\
            -self.fe.across_edge_deln_matrix['bottom']
          h_minus = self.fe.width_cell[1]
          h_plus = self.fe.width_cell[1]
          next_cell_offset = self.param.n_x
          
        dof = self.edge_index(0,edge,inside)
        D_minus,D_plus = self.compute_diffusion_coefficient(dof,edge_offset)
        K = self.compute_penalty_coefficient(dof,edge_offset,D_minus,D_plus,
            h_minus,h_plus,inside)

        for i in xrange(0,2) :
          i_edge_pos = self.edge_index(i,edge,inside)
          for j in xrange(0,2) :
            j_edge_pos = self.edge_index(j,edge,inside)

# First edge term
            x[i_edge_pos] += K*edge_mass_matrix[i,j]*x_krylov[j_edge_pos]
            x[i_edge_pos] -= K*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+edge_offset]
            x[i_edge_pos+edge_offset] += K*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+edge_offset]
            x[i_edge_pos+edge_offset] -= K*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos]

# Internal terms (-,-)
        cell = np.floor(self.edge_index(0,edge,inside)/self.fe.n_dofs_per_cell)
        for i in xrange(0,4) :
          i_pos = self.index(i,cell)
          for j in xrange(0,4) :
            j_pos = self.index(j,cell)

            x[i_pos] -= D_minus/2.*edge_deln_matrix[i,j]*x_krylov[j_pos]
            x[i_pos] -= D_minus/2.*edge_deln_matrix[j,i]*x_krylov[j_pos]

# External terms (+,+)
        next_cell = cell+next_cell_offset
        for i in xrange(0,4) :
          i_pos = self.index(i,next_cell)
          for j in xrange(0,4) :
            j_pos = self.index(j,next_cell)

            x[i_pos] += D_plus/2.*outside_edge_deln_matrix[i,j]*x_krylov[j_pos]
            x[i_pos] += D_plus/2.*outside_edge_deln_matrix[j,i]*x_krylov[j_pos]

# Mixte terms (+,-) 
        for i in xrange(0,4) :
          i_pos = self.index(i,cell)
          for j in xrange(0,4) :
            j_pos = self.index(j,next_cell)

            x[i_pos] += D_minus/2.*in_across_edge_deln_matrix[i,j]*\
                x_krylov[j_pos]
            x[i_pos] -= D_plus/2.*in_across_edge_deln_matrix[j,i]*\
                x_krylov[j_pos]

## Mixte terms (-,+) 
        for i in xrange(0,4) :
          i_pos = self.index(i,next_cell)
          for j in xrange(0,4) :
            j_pos = self.index(j,cell)

            x[i_pos] -= D_plus/2.*out_across_edge_deln_matrix[i,j]*\
              x_krylov[j_pos]
            x[i_pos] += D_minus/2.*out_across_edge_deln_matrix[j,i]*\
              x_krylov[j_pos]
      else :
        is_vertical = self.compute_vertical(edge,inside)
        Jdotn = self.compute_Jdotn(edge,is_vertical)
        if is_vertical==True :
          edge_mass_matrix = self.fe.vertical_edge_mass_matrix
          h_minus = self.fe.width_cell[0]
          h_plus = 0.
          if Jdotn>0. :
            edge_deln_matrix = self.fe.edge_deln_matrix['right']
          else :
            edge_deln_matrix = self.fe.edge_deln_matrix['left']
        else :
          edge_mass_matrix = self.fe.horizontal_edge_mass_matrix
          h_minus = self.fe.width_cell[1]
          h_plus = 0.
          if Jdotn>0. :
            edge_deln_matrix = self.fe.edge_deln_matrix['top']
          else :
            edge_deln_matrix = self.fe.edge_deln_matrix['bottom']

        edge_offset = 0
        dof = self.edge_index(0,edge,inside)
        D_minus,D_plus = self.compute_diffusion_coefficient(dof,edge_offset)
        K = self.compute_penalty_coefficient(dof,edge_offset,D_minus,D_plus,
            h_minus,h_plus,inside)

# First edge term
        for i in xrange(0,2) :
          i_edge_pos = self.edge_index(i,edge,inside)
          for j in xrange(0,2) :
            j_edge_pos = self.edge_index(j,edge,inside)

            x[i_edge_pos] += K*edge_mass_matrix[i,j]*x_krylov[j_edge_pos]
        
        cell = int(np.floor(self.edge_index(0,edge,inside)/self.fe.n_dofs_per_cell))
        for i in xrange(0,4) :
          i_pos = self.index(i,cell)
          for j in xrange(0,4) :
            j_pos = self.index(j,cell)
          
# Second edge term
            x[i_pos] -= 0.5*D_minus*edge_deln_matrix[i,j]*x_krylov[j_pos]

# Third edge term
            x[i_pos] -= 0.5*D_minus*edge_deln_matrix[j,i]*x_krylov[j_pos]

    return x

#----------------------------------------------------------------------------#

  def build_lhs(self) :
    """Build the matrix of the MIP that will be used by CG and the algebraic
    multigrid method."""

#    A = scipy.sparse.lil_matrix((4*self.param.n_cells,4*self.param.n_cells))
#    x = np.zeros([4*self.param.n_cells,4*self.param.n_cells])
    x = scipy.sparse.lil_matrix((self.size,self.size))

    for cell in xrange(0,self.param.n_cells) :
      i,j = self.cell_mapping(cell) 
      i_mat = self.param.mat_id[i,j]
      sig_a = self.param.sig_t[i_mat]-self.param.sig_s[0,i_mat] 
# Compute the diffusion coefficient D
      if self.param.sig_s.shape[0]>1 :
        sig_tr = self.param.sig_t[i_mat]-self.param.sig_s[1,i_mat]
      else :
        sig_tr = self.param.sig_t[i_mat]
      D = 1./(3.*sig_tr)

      for i in xrange(0,self.fe.n_dofs_per_cell) :
        pos_i = self.index(i,cell)
        for j in xrange(0,self.fe.n_dofs_per_cell) :
          pos_j = self.index(j,cell)

          x[pos_i,pos_j] += sig_a*self.fe.mass_matrix[i,j]
          x[pos_i,pos_j] += D*self.fe.stiffness_matrix[i,j]

# The edges are numbered as follow : first the vertical ones (left to right
# and then the bottom to top) then the horizontal ones (bottom to top and left
# to right).
#     4
#   -----
#   |   |
# 1 |   | 2
#   |   | 
#   -----
#     3

    n_edges = self.param.n_x*(self.param.n_y+1)+self.param.n_y*\
        (self.param.n_x+1)
    for edge in xrange(0,n_edges) :
      inside = self.interior(edge)
      if inside == True :
        is_vertical = self.compute_vertical(edge,inside) 
        if is_vertical==True :
          edge_offset = self.compute_edge_offset('right')
          edge_mass_matrix = self.fe.vertical_edge_mass_matrix
          edge_deln_matrix = self.fe.edge_deln_matrix['right']
          outside_edge_deln_matrix = -self.fe.edge_deln_matrix['left']
          in_across_edge_deln_matrix = self.fe.across_edge_deln_matrix['right']
          out_across_edge_deln_matrix =\
              -self.fe.across_edge_deln_matrix['left']
          h_minus = self.fe.width_cell[0]
          h_plus = self.fe.width_cell[0]
          next_cell_offset = 1
        else :
          edge_offset = self.compute_edge_offset('top')
          edge_mass_matrix = self.fe.horizontal_edge_mass_matrix
          edge_deln_matrix = self.fe.edge_deln_matrix['top']
          outside_edge_deln_matrix = -self.fe.edge_deln_matrix['bottom']
          in_across_edge_deln_matrix = self.fe.across_edge_deln_matrix['top']
          out_across_edge_deln_matrix =\
            -self.fe.across_edge_deln_matrix['bottom']
          h_minus = self.fe.width_cell[1]
          h_plus = self.fe.width_cell[1]
          next_cell_offset = self.param.n_x
          
        dof = self.edge_index(0,edge,inside)
        D_minus,D_plus = self.compute_diffusion_coefficient(dof,edge_offset)
        K = self.compute_penalty_coefficient(dof,edge_offset,D_minus,D_plus,
            h_minus,h_plus,inside)

        for i in xrange(0,2) :
          i_edge_pos = self.edge_index(i,edge,inside)
          for j in xrange(0,2) :
            j_edge_pos = self.edge_index(j,edge,inside)

# First edge term
            x[i_edge_pos,j_edge_pos] += K*edge_mass_matrix[i,j]
            x[i_edge_pos,j_edge_pos+edge_offset] -= K*edge_mass_matrix[i,j]
            x[i_edge_pos+edge_offset,j_edge_pos+edge_offset] +=\
                K*edge_mass_matrix[i,j]
            x[i_edge_pos+edge_offset,j_edge_pos] -= K*edge_mass_matrix[i,j]

# Internal terms (-,-)
        cell = np.floor(self.edge_index(0,edge,inside)/self.fe.n_dofs_per_cell)
        for i in xrange(0,4) :
          i_pos = self.index(i,cell)
          for j in xrange(0,4) :
            j_pos = self.index(j,cell)

            x[i_pos,j_pos] -= D_minus/2.*edge_deln_matrix[i,j]
            x[i_pos,j_pos] -= D_minus/2.*edge_deln_matrix[j,i]

# External terms (+,+)
        next_cell = cell+next_cell_offset
        for i in xrange(0,4) :
          i_pos = self.index(i,next_cell)
          for j in xrange(0,4) :
            j_pos = self.index(j,next_cell)

            x[i_pos,j_pos] += D_plus/2.*outside_edge_deln_matrix[i,j]
            x[i_pos,j_pos] += D_plus/2.*outside_edge_deln_matrix[j,i]

# Mixte terms (+,-) 
        for i in xrange(0,4) :
          i_pos = self.index(i,cell)
          for j in xrange(0,4) :
            j_pos = self.index(j,next_cell)
  
            x[i_pos,j_pos] += D_minus/2.*out_across_edge_deln_matrix[i,j]
            x[i_pos,j_pos] -= D_plus/2.*out_across_edge_deln_matrix[j,i]

# Mixte terms (-,+) 
        for i in xrange(0,4) :
          i_pos = self.index(i,next_cell)
          for j in xrange(0,4) :
            j_pos = self.index(j,cell)

            x[i_pos,j_pos] -= D_plus/2.*in_across_edge_deln_matrix[i,j]
            x[i_pos,j_pos] += D_minus/2.*in_across_edge_deln_matrix[j,i]
      else :
        is_vertical = self.compute_vertical(edge,inside)
        Jdotn = self.compute_Jdotn(edge,is_vertical)
        if is_vertical==True :
          edge_mass_matrix = self.fe.vertical_edge_mass_matrix
          h_minus = self.fe.width_cell[0]
          h_plus = 0.
          if Jdotn>0. :
            edge_deln_matrix = self.fe.edge_deln_matrix['right']
          else :
            edge_deln_matrix = self.fe.edge_deln_matrix['left']
        else :
          edge_mass_matrix = self.fe.horizontal_edge_mass_matrix
          h_minus = self.fe.width_cell[1]
          h_plus = 0.
          if Jdotn>0. :
            edge_deln_matrix = self.fe.edge_deln_matrix['top']
          else :
            edge_deln_matrix = self.fe.edge_deln_matrix['bottom']

        edge_offset = 0
        dof = self.edge_index(0,edge,inside)
        D_minus,D_plus = self.compute_diffusion_coefficient(dof,edge_offset)
        K = self.compute_penalty_coefficient(dof,edge_offset,D_minus,D_plus,
            h_minus,h_plus,inside)

# First edge term
        for i in xrange(0,2) :
          i_edge_pos = self.edge_index(i,edge,inside)
          for j in xrange(0,2) :
            j_edge_pos = self.edge_index(j,edge,inside)

            x[i_edge_pos,j_edge_pos] += K*edge_mass_matrix[i,j]
        
        cell = int(np.floor(self.edge_index(0,edge,inside)/self.fe.n_dofs_per_cell))
        for i in xrange(0,4) :
          i_pos = self.index(i,cell)
          for j in xrange(0,4) :
            j_pos = self.index(j,cell)
          
# Second edge term
            x[i_pos,j_pos] -= 0.5*D_minus*edge_deln_matrix[i,j]

# Third edge term
            x[i_pos,j_pos] -= 0.5*D_minus*edge_deln_matrix[j,i]
        
    return x
