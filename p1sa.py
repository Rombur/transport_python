# Python code
# Author: Bruno Turcksin
# Date: 2011-04-05 09:36:14.092927

#----------------------------------------------------------------------------#
## Class p1sa                                                               ##
#----------------------------------------------------------------------------#

"""Solve the P1SA equation"""

import numpy as np
import scipy.sparse.linalg
import synthetic_acceleration as na

class p1sa(na.synthetic_acceleration)  :
  """Preconditioner for the transport equation using the P1SA equation."""

  def __init__(self,parameters,quadrature,fe,tol) :

    super(p1sa,self).__init__(parameters,quadrature,fe,tol)

#----------------------------------------------------------------------------#

  def solve(self,x) :
    """Solve the P1SA equation where the right-hand-side is a Krylov vector."""

# Restrict the Krylov vector to build the rhs
    size = int(3*4*self.param.n_cells)
    self.p1sa_b = np.zeros([size])
    self.krylov_vector = np.zeros([size])
    x_size = x.shape
    if x_size[0] == 4*self.param.n_cells :
      self.krylov_vector[0:4*self.param.n_cells] = x[:]
      size_output = 4*self.param.n_cells
    else :
      self.krylov_vector[:] = x[0:size]
      size_output = size

# Compute the right-hand-side
    self.compute_rhs()

# GMRES solver 
    A = scipy.sparse.linalg.LinearOperator((size,size),matvec=self.mv,
        rmatvec=None,dtype=float)
    flux,flag = scipy.sparse.linalg.bicgstab(A,self.p1sa_b,tol=self.tol)

    if flag!=0 :
      print 'P1SA did not converge.'

# Project the P1SA solution on the whole space
    krylov_space_size = x.shape[0]
    output = np.zeros([krylov_space_size])
    output[0:size_output] = flux[0:size_output]

    return output

#----------------------------------------------------------------------------#

  def compute_rhs(self) :
    """Compute the rhs of the P1SA equation."""

    n_dofs = 4
    x_current_offset = n_dofs*self.param.n_cells
    y_current_offset = 2*n_dofs*self.param.n_cells

    for cell in xrange(0,self.param.n_cells) :
      for i in xrange(0,n_dofs) :
        pos_i = self.index(i,cell)
        for j in xrange(0,n_dofs) :
          pos_j = self.index(j,cell)
          self.p1sa_b[pos_i] += self.fe.mass_matrix[i,j]*\
              self.krylov_vector[pos_j]
          self.p1sa_b[pos_i+x_current_offset] += 3.*self.fe.mass_matrix[i,j]*\
              self.krylov_vector[pos_j+x_current_offset]
          self.p1sa_b[pos_i+y_current_offset] += 3.*self.fe.mass_matrix[i,j]*\
              self.krylov_vector[pos_j+y_current_offset]

#----------------------------------------------------------------------------#

  def mv(self,x_krylov) :
    """Perform the matrix-vector multiplication needed by BICGSTAB."""

    boundary = 'vacuum'
    x = np.zeros([3*4*self.param.n_cells])
    x_current_offset = 4*self.param.n_cells
    y_current_offset = 2*4*self.param.n_cells

    for cell in xrange(0,self.param.n_cells) :
      
      i,j = self.cell_mapping(cell)
      i_mat = self.param.mat_id[i,j]
      sig_a = self.param.sig_t[i_mat]-self.param.sig_s[0,i_mat]
      if self.param.sig_s.shape[0]>1 :
        sig_tr = self.param.sig_t[i_mat]-self.param.sig_s[1,i_mat]
      else :
        sig_tr = self.param.sig_t[i_mat]

      for i in xrange(0,4) :
        pos_i = self.index(i,cell)
        for j in xrange(0,4) :
          pos_j = self.index(j,cell)
          
          x[pos_i] += sig_a*self.fe.mass_matrix[i,j]*x_krylov[pos_j]

          x[pos_i+x_current_offset] += 3*sig_tr*self.fe.mass_matrix[i,j]*\
              x_krylov[pos_j+x_current_offset]
          x[pos_i+y_current_offset] += 3*sig_tr*self.fe.mass_matrix[i,j]*\
              x_krylov[pos_j+y_current_offset]

          x[pos_i+x_current_offset] += self.fe.x_grad_matrix[j,i]*\
              x_krylov[pos_j]
          x[pos_i+y_current_offset] += self.fe.y_grad_matrix[j,i]*\
              x_krylov[pos_j]

          x[pos_i] -= self.fe.x_grad_matrix[i,j]*\
              x_krylov[pos_j+x_current_offset]     
          x[pos_i] -= self.fe.y_grad_matrix[i,j]*\
              x_krylov[pos_j+y_current_offset]     
          
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
          current_offset = x_current_offset
          tan_current_offset = y_current_offset
          edge_mass_matrix = self.fe.vertical_edge_mass_matrix
        else :
          edge_offset = self.compute_edge_offset('top')
          current_offset = y_current_offset
          tan_current_offset = x_current_offset
          edge_mass_matrix = self.fe.horizontal_edge_mass_matrix
# Need to go to the right or to the top cell -> offset is different
        right_edge_offset = self.compute_edge_offset('right')
        top_edge_offset = self.compute_edge_offset('top')

        for i in xrange(0,2) :
          i_edge_pos = self.edge_index(i,edge,inside)
          for j in xrange(0,2) :
            j_edge_pos = self.edge_index(j,edge,inside) 

# First edge term
            x[i_edge_pos] += 0.25*edge_mass_matrix[i,j]*x_krylov[j_edge_pos]
            x[i_edge_pos] -= 0.25*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+edge_offset]
            x[i_edge_pos+edge_offset] -= 0.25*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos]
            x[i_edge_pos+edge_offset] += 0.25*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+edge_offset]

# Second edge term
            x[i_edge_pos+edge_offset+current_offset] += 0.5*\
                edge_mass_matrix[i,j]*x_krylov[j_edge_pos+edge_offset]
            x[i_edge_pos+current_offset] += 0.5*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+edge_offset]
            x[i_edge_pos+edge_offset+current_offset] -= 0.5*\
                edge_mass_matrix[i,j]*x_krylov[j_edge_pos]
            x[i_edge_pos+current_offset] -= 0.5*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos]

# Third edge term 
            x[i_edge_pos+edge_offset] -= 0.5*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+edge_offset+current_offset]
            x[i_edge_pos] += 0.5*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+edge_offset+current_offset]
            x[i_edge_pos+edge_offset] -= 0.5*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+current_offset]
            x[i_edge_pos] += 0.5*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+current_offset]

# Fourth (J dot n jump) and part of the fifth (normal jump) edge terms put
# together
            x[i_edge_pos+edge_offset+current_offset] += 9./8.*\
                edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+edge_offset+current_offset]
            x[i_edge_pos+current_offset] -= 9./8.*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+edge_offset+current_offset]
            x[i_edge_pos+edge_offset+current_offset] -= 9./8.*\
                edge_mass_matrix[i,j]*x_krylov[j_edge_pos+current_offset]
            x[i_edge_pos+current_offset] += 9./8.*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+current_offset]

# Tangential jump (fifth edge term)
            x[i_edge_pos+edge_offset+tan_current_offset] += 9./16.*\
                edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+edge_offset+tan_current_offset]
            x[i_edge_pos+tan_current_offset] -= 9./16.*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+edge_offset+tan_current_offset]
            x[i_edge_pos+edge_offset+tan_current_offset] -= 9./16.*\
                edge_mass_matrix[i,j]*x_krylov[j_edge_pos+tan_current_offset]
            x[i_edge_pos+tan_current_offset] += 9./16.*edge_mass_matrix[i,j]*\
                x_krylov[j_edge_pos+tan_current_offset]
      else :
# Caveat : the normal is always outgoing on the boundary. It is not the same
# rule than for the interior edges.
        if boundary=='vacuum' :
          is_vertical = self.compute_vertical(edge,inside)
          if is_vertical==True :
            current_offset = x_current_offset
            tan_current_offset = y_current_offset
            edge_mass_matrix = self.fe.vertical_edge_mass_matrix
          else :
            current_offset = y_current_offset
            tan_current_offset = x_current_offset
            edge_mass_matrix = self.fe.horizontal_edge_mass_matrix
          for i in xrange(0,2) :
            i_edge_pos = self.edge_index(i,edge,inside) 
            
            Jdotn = self.compute_Jdotn(edge,is_vertical)

            for j in xrange(0,2) :
              j_edge_pos = self.edge_index(j,edge,inside)

              x[i_edge_pos] += 0.25*edge_mass_matrix[i,j]*x_krylov[j_edge_pos]
              x[i_edge_pos+current_offset] -= 0.5*Jdotn*edge_mass_matrix[i,j]*\
                  x_krylov[j_edge_pos]
              x[i_edge_pos] += 0.5*Jdotn*edge_mass_matrix[i,j]*\
                  x_krylov[j_edge_pos+current_offset]
              x[i_edge_pos+tan_current_offset] += 9./16.*\
                  edge_mass_matrix[i,j]*x_krylov[j_edge_pos+tan_current_offset]
# (Jdotn)^2 -> always positive -> omitted
              x[i_edge_pos+current_offset] += 9./8.*edge_mass_matrix[i,j]*\
                  x_krylov[j_edge_pos+current_offset]
        else :
          is_vertical = self.compute_vertical(edge,inside) 
          if is_vertical==True :
            current_offset = x_current_offset
            edge_mass_matrix = self.fe.vertical_edge_mass_matrix
          else :
            current_offset = y_current_offset
            edge_mass_matrix = self.fe.horizontal_edge_mass_matrix
          for i in xrange(0,2) :
            i_edge_pos = self.edge_index(i,edge,inside)
            for j in xrange(0,2) :
              j_edge_pos = self.edge_index(j,edge,inside)
              x[i_edge_pos+current_offset] += 9./4.*edge_mass_matrix[i,j]*\
                  x_krylov[j_edge_pos+current_offset]

    return x              
