# Python code
# Author: Bruno Turcksin
# Date: 2011-03-31 11:34:16.287390

#----------------------------------------------------------------------------#
## Class parameters                                                         ##
#----------------------------------------------------------------------------#

"""Read the inputs for the transport code"""

import numpy as np

class parameters  :
  """Read the inputs (Sn order, material properties and geometry) for the
    transport code."""

  def __init__(self,galerkin,fokker_planck,TC,optimal,is_precond,
      multigrid,level,L_max,sn) :
# geometry
    self.mat_id = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],\
        [0,0,0,0,0]])
    self.src_id = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],\
        [0,0,0,0,0]])
    self.src = np.array([10.])
    self.width = np.array([1.,1.])
    self.n_div = np.array([2,2])
    size = self.mat_id.shape
    self.n_x = self.n_div[0]*size[0]
    self.n_y = self.n_div[1]*size[1]
    self.width_x_cell = self.width[0]/self.n_div[0]
    self.width_y_cell = self.width[1]/self.n_div[1]
    self.n_cells = self.n_x*self.n_y
    self.inc_left = np.array([10.,10.,10.,10.,10.])
    self.inc_right = np.array([0,0,0,0,0])
    self.inc_top = np.array([0,0,0,0,0])
    self.inc_bottom = np.array([0,0,0,0,0])
    self.resize()
# material property
    self.L_max = L_max
    self.sig_t = np.array([37.])
    self.fokker_planck = fokker_planck
    if fokker_planck == False :
      self.sig_s = np.zeros((3,1))
      self.sig_s[0,0] = 0.99
    else :
      self.alpha = 1
      self.level = level
      self.fokker_planck_xs()
    self.n_mom = self.sig_s.shape[0]
# Sn property
    self.galerkin = galerkin
    if self.galerkin == False :
      self.sn = sn
    else :
      self.sn = self.L_max
    self.TC = TC
    self.optimal = optimal
    if TC == True :
        self.transport_correction()
    self.is_precond = is_precond
    self.multigrid = multigrid

#----------------------------------------------------------------------------#

  def resize(self) :
    """Resize the vector because of the refinement."""

    new_i_size = self.mat_id.shape[0]*self.n_div[0]
    new_j_size = self.mat_id.shape[1]*self.n_div[1]

    mat_id_tmp = np.zeros([new_i_size,new_j_size])
    src_id_tmp = np.zeros([new_i_size,new_j_size])
    inc_left_tmp = np.zeros([new_i_size])
    inc_right_tmp = np.zeros([new_i_size])
    inc_top_tmp = np.zeros([new_j_size])
    inc_bottom_tmp = np.zeros([new_j_size])

    for i in xrange(0,new_i_size) :
      old_i = int(i)/int(self.n_div[0])
      inc_left_tmp[i] = self.inc_left[old_i]
      inc_right_tmp[i] = self.inc_right[old_i]
      for j in xrange(0,new_j_size) :
        old_j = int(j)/int(self.n_div[1]) 
        mat_id_tmp[i,j] = self.mat_id[old_i,old_j]
        src_id_tmp[i,j] = self.src_id[old_i,old_j]
    for j in xrange(0,new_j_size) :
      old_j = int(j)/int(new_j_size)
      inc_bottom_tmp[j] = self.inc_bottom[old_j]
      inc_top_tmp[j] = self.inc_top[old_j]

    self.mat_id = mat_id_tmp.transpose()
    self.src_id = src_id_tmp.transpose()
    self.inc_left = inc_left_tmp
    self.inc_right = inc_right_tmp
    self.inc_top = inc_top_tmp
    self.inc_bottom = inc_bottom_tmp

#----------------------------------------------------------------------------#

  def fokker_planck_xs(self) :
    """Compute the Fokker-Planck cross sections."""

    size = self.L_max*(self.L_max+2)/2
    self.sig_s = np.zeros((size,1))
   
    pos = 0
# Compute the effective L_max used by the angular multigrid
    L_max_eff = self.L_max+self.level*self.L_max
    for i in xrange(0,self.L_max) :
      for j in xrange(0,i+1) :
        self.sig_s[pos] = self.alpha/2.0*(L_max_eff*(L_max_eff+1)-i*(i+1))
        pos += 1
    if self.level==1 :
      for i in xrange(pos,size) :
        self.sig_s[i] = self.alpha/2.0*(L_max_eff*(L_max_eff+1)-self.L_max*\
            (self.L_max+1))

#----------------------------------------------------------------------------#

  def transport_correction(self) :
    """Compute the transport correction for the cross sections."""

    if self.optimal == True :
      correction = (self.sig_s[3]+self.sig_s[-1])/2.
    else :
      correction = self.sig_s[1]
    
    self.sig_t -= correction
    self.sig_s -= correction
