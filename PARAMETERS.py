# Python code
# Author: Bruno Turcksin
# Date: 2011-03-31 11:34:16.287390

#----------------------------------------------------------------------------#
## Class PARAMETERS                                                         ##
#----------------------------------------------------------------------------#

"""Read the inputs for the transport code"""

import numpy as np

class PARAMETERS(object) :
  """Read the inputs (Sn order, material properties and geometry) for the
    transport code."""

  def __init__(self,galerkin,L_max,sn,print_to_file,gmres,x_width,y_width,
      n_x_div,n_y_div,inc_bottom,inc_right,inc_top,inc_left,mat_id,src_id, 
      src,sig_t,sig_s,weight,verbose,discretization) :

    super(PARAMETERS,self).__init__()
# geometry
    self.x_width = x_width
    self.y_width = y_width
    self.n_x_div = n_x_div
    self.n_y_div = n_y_div
    self.inc_left = inc_left
    self.inc_right = inc_right
    self.inc_top = inc_top
    self.inc_bottom = inc_bottom
    self.mat_id = mat_id
    self.src_id = src_id
    self.src = src
    self.n_x = self.n_x_div.sum()
    self.n_y = self.n_y_div.sum()
    self.n_cells = self.n_x*self.n_y
    self.Resize()
# material property
    self.L_max = int(L_max)
    self.sig_t = sig_t
    self.sig_s = sig_s
    for i_mat in range(0,int(self.mat_id.max()+1)) :
      if self.sig_s[i_mat,0]>self.sig_t[i_mat] :
         raise NameError('sig_s[0] is greater than sig_t.')
    self.n_mom = self.sig_s.shape[0]
    self.galerkin = galerkin
    if self.galerkin == False :
      self.sn = int(sn)
    else :
      self.sn = self.L_max
    self.weight = weight
    self.verbose = verbose
    self.print_to_file = print_to_file
    self.gmres = gmres
    self.discretization = discretization

#----------------------------------------------------------------------------#

  def Resize(self) :
    """Resize the vector because of the refinement."""

    mat_id_tmp = np.zeros([self.n_x,self.n_y])
    src_id_tmp = np.zeros([self.n_x,self.n_y])
    x_width_tmp = np.zeros([self.n_x])
    y_width_tmp = np.zeros([self.n_y])

    pos_x = 0
    old_pos_x = 0
    for x_div in self.n_x_div :
      for i in range(x_div) :
        x_width_tmp[pos_x] = self.x_width[old_pos_x]/x_div
        pos_y = 0
        old_pos_y = 0
        for y_div in self.n_y_div :
          for j in range(y_div) :
            mat_id_tmp[pos_x,pos_y] = self.mat_id[old_pos_x,old_pos_y]
            src_id_tmp[pos_x,pos_y] = self.src_id[old_pos_x,old_pos_y]
            pos_y += 1
          old_pos_y += 1
        pos_x += 1
      old_pos_x += 1  
    pos_y = 0
    old_pos_y = 0
    for y_div in self.n_y_div :
      for i in range(y_div) :
        y_width_tmp[pos_y] = self.y_width[old_pos_y]/y_div
        pos_y += 1
      old_pos_y += 1

    self.mat_id = mat_id_tmp
    self.src_id = src_id_tmp
    self.x_width = x_width_tmp
    self.y_width = y_width_tmp
