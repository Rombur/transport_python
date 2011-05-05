# Python code
# Author: Bruno Turcksin
# Date: 2011-04-18 18:37:34.846393

#----------------------------------------------------------------------------#
## Class synthetic_acceleration                                             ##
#----------------------------------------------------------------------------#

"""Solve the synthetic preconditioner"""

import numpy as np

class synthetic_acceleration(object) :
  """This class is the base class for the synthetic acceleration : DSA and
  P1SA."""

  def __init__(self,parameters,fe,tol,output_file) :
    self.param = parameters
    self.fe = fe
    self.tol = tol
    self.output_file = output_file
    
#----------------------------------------------------------------------------#

  def solve(self,x) :
    """Solve the SA equation where the right-hand-side is a Krylov vector. The
    function is purely virtual."""

    raise NotImplementedError("solve is purely virtual and must be overriden.")

#----------------------------------------------------------------------------#

  def compute_rhs(self) :
    """Compute the rhs of the SA equation. The function is purely virtual."""

    raise NotImplementedError("compute_rhs is purely virtual and must be overridden.")

#----------------------------------------------------------------------------#

  def mv(self,x_krylov) :
    """Perform the matrix-vector multiplication needed by the Krylov  method."""

    raise NotImplementedError("mv is purely virtual and must be overriden.")

#----------------------------------------------------------------------------#

  def index(self,i,cell) :
    """Compute the position in the (scalar flux) vector knowing the cell and 
    the dof."""

    return  i+4*cell

#----------------------------------------------------------------------------#

  def cell_mapping(self,cell) :
    """Get the i,j pair for a given a cell."""

    j = np.floor(cell/self.param.n_x)
    i = cell - j*self.param.n_x

    return i,j

#----------------------------------------------------------------------------#

  def interior(self,edge) :
    """Return True if the edge is inside the domain and False if the edge is
    on the boundary."""

    if edge<self.param.n_y*(self.param.n_x+1) :
      modulo = np.fmod(edge,self.param.n_x+1)
      if modulo==0 or modulo==self.param.n_x :
        inside = False
      else :
        inside = True
    else :
      modulo = np.fmod(edge-(self.param.n_x+1),self.param.n_y+1)
      if modulo==0 or modulo==self.param.n_y :
        inside = False
      else :
        inside = True
    
    return  inside

#----------------------------------------------------------------------------#

  def compute_vertical(self,edge,interior) :
    """Return a boolean to True if the edge is vertical and False
    otherwise."""

    if interior==True :
      if edge<self.param.n_y*(self.param.n_x+1) :
        is_vertical = True
      else :
        is_vertical = False
    else :
      if edge<self.param.n_y*(self.param.n_x+1) :
        is_vertical = True
      else :
        is_vertical = False

    return is_vertical

#----------------------------------------------------------------------------#

  def edge_index(self,i,edge,interior) :
    """Compute the index of an edge."""

    if interior==True :
      if edge < self.param.n_y*(self.param.n_x+1) :
        line = np.floor(edge/(self.param.n_x+1))
        column = edge-line*(self.param.n_x+1)-1
        index = 4*self.param.n_x*line+4*column+2+i
      else :
        edge = edge-self.param.n_y*(self.param.n_x+1)
        column = np.floor(edge/(self.param.n_y+1))
        line = edge-column*(self.param.n_y+1)-1
        index = int(4*self.param.n_x*line+4*column+1+2*i)
    else :
      if edge < self.param.n_y*(self.param.n_x+1) :
        if np.fmod(edge,self.param.n_x+1)==0 :
          index = 4*self.param.n_x*edge/(self.param.n_x+1)+i
        else :
          index = 4*self.param.n_x*(edge+1)/(self.param.n_x+1)-2+i
      else :
        edge = edge-self.param.n_y*(self.param.n_x+1)
        column = np.floor(edge/(self.param.n_y+1))
        if np.fmod(edge,self.param.n_y+1)==0 :
          index = 4*column+2*i
        else :
          index = int(4*self.param.n_x*(self.param.n_y-1)+4*column+1+2*i)
    
    return index

#----------------------------------------------------------------------------#

  def compute_edge_offset(self,next_cell) :
    """Compute the offset for the right and top cell."""

    if next_cell=='right' :
      offset = 2
    else :
      if next_cell=='top' :
        offset = 4*self.param.n_x-1
      else :
        self.print_message('Illegal cell.')

    return offset

#----------------------------------------------------------------------------#

  def compute_Jdotn(self,edge,is_vertical) :
    """Compute the dot product between the current and the normal."""

    if is_vertical==True :
      if np.fmod(edge,self.param.n_x+1)==0 :
        Jdotn = -1.
      else :
        Jdotn = 1.
    else :
      if np.fmod(edge-self.param.n_y*(self.param.n_x+1),self.param.n_y+1)==0 :
        Jdotn = -1.
      else :
        Jdotn = 1.

    return Jdotn    

#----------------------------------------------------------------------------#

  def print_message(self,a) :
    """Print the given message a on the screen or in a file."""

    if self.param.print_to_file==True :
      self.output_file.write(a+'\n')
    else :
      print a 
