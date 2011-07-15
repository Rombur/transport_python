# Python code
# Author: Bruno Turcksin
# Date: 2011-03-31 11:34:16.287390

#----------------------------------------------------------------------------#
## Class parameters                                                         ##
#----------------------------------------------------------------------------#

"""Read the inputs for the transport code"""

import numpy as np
import utils

class parameters(object) :
  """Read the inputs (Sn order, material properties and geometry) for the
    transport code."""

  def __init__(self,galerkin,fokker_planck,TC,optimal,preconditioner,
      multigrid,L_max,sn,level=0,max_level=0) :

    super(parameters,self).__init__()
# geometry
    self.mat_id = np.array([[0]])
    self.src_id = np.array([[0]])
    self.src = np.array([0.])
    self.width = np.array([5.,5.])
    self.n_div = np.array([2,2])
    size = self.mat_id.shape
    self.n_x = self.n_div[0]*size[0]
    self.n_y = self.n_div[1]*size[1]
    self.width_x_cell = self.width[0]/self.n_div[0]
    self.width_y_cell = self.width[1]/self.n_div[1]
    self.n_cells = self.n_x*self.n_y
    self.inc_left = np.array([0.])
    if self.inc_left.shape[0]!=self.mat_id.shape[0] :
      utils.abort('inc_left has a wrong size.')
    self.inc_right = np.array([10.])
    if self.inc_right.shape[0]!=self.mat_id.shape[0] :
      utils.abort('inc_right has a wrong size.')
    self.inc_top = np.array([0.])
    if self.inc_top.shape[0]!=self.mat_id.shape[1] :
      utils.abort('inc_top has a wrong size.')
    self.inc_bottom = np.array([0.])
    if self.inc_bottom.shape[0]!=self.mat_id.shape[1] :
      utils.abort('inc_bottom has a wrong size.')
    self.resize()
# material property
    self.alpha = np.array([1.])
    self.level = level
    self.max_level = max_level
    if level==0 :
      tmp_sn = sn
      if preconditioner=='P1SA':
        while tmp_sn>4 :
          self.max_level += 1
          tmp_sn /= 2
      else :
        while tmp_sn>2 :
          self.max_level += 1
          tmp_sn /= 2
    self.L_max = L_max
    #self.sig_t = np.array([10.])
    self.sig_t = np.array([36.])
    #self.sig_t = np.array([78.])
    #self.sig_t = np.array([136.])
    self.fokker_planck = fokker_planck
    if fokker_planck == False :
      self.sig_s = np.zeros((sn*(sn+2)/2,1))
      if sn==1 :
        self.sig_s = np.zeros((4,1))
      self.sig_s[0,0] = 100.
    else :
      if sn==1 :
        self.L_max = 2
        self.level -= 1
        size = self.L_max*(self.L_max+2)/2
        self.sig_s = np.zeros((size,self.alpha.shape[0]))
        self.fokker_planck_xs(size)
        self.L_max = 1
        self.level += 1
      else :
        size = self.L_max*(self.L_max+2)/2
        self.sig_s = np.zeros((size,self.alpha.shape[0]))
        self.fokker_planck_xs(size)
    for i_mat in xrange(0,self.alpha.shape[0]) :
      if self.sig_s[0,i_mat]>self.sig_t[i_mat] :
        utils.abort('sig_s[0] is greater than sig_t.')
    self.n_mom = self.sig_s.shape[0]
# Sn property
    self.multigrid = multigrid
    self.galerkin = galerkin
# Solver properties
    self.preconditioner = preconditioner
# Projection on the scalar flux (scalar) or the scalar flux + current (current)
    self.projection = 'scalar'
    if self.galerkin == False :
      self.sn = sn
    else :
      self.sn = self.L_max
    self.TC = TC
    self.optimal = optimal
    if TC == True :
        self.transport_correction()
# If matrix_free is True, the preconditioner matrix is not build
    self.matrix_free = False
# If the matrix is build and pyamg is True, the preconditioner is solve using
# a algebraic multigrid method
    self.pyamg = False
# Maximum number of v-cycle allowed to solve MIP
    self.v_cycle = 50
# If MIP is solved using CG and my_cg is True, then my own implementation of
# CG is used. Otherwise the CG of scipy is used
    self.my_cg = True
# If a multigrid method is used and accel is True, the multigrid is used to
# accelerate a Krylov solver. Otherwise the multigrid is used in standalone
    self.accel = False
    self.verbose = 2
# If print_to_file is True, the message are written on a file, otherwise they
# are printed on the scree
    self.print_to_file = False
# Toggle between SI and GMRES when multigrid is not use
    self.gmres = True
# Compute the solution of the P1SA and MiP problems after solving the
# transport problem.
    self.full_output = False

#----------------------------------------------------------------------------#

  def resize(self) :
    """Resize the vector because of the refinement."""

    new_i_size = self.mat_id.shape[0]*self.n_div[0]
    new_j_size = self.mat_id.shape[1]*self.n_div[1]

    mat_id_tmp = np.zeros([new_i_size,new_j_size])
    src_id_tmp = np.zeros([new_i_size,new_j_size])
    inc_left_tmp = np.zeros([new_j_size])
    inc_right_tmp = np.zeros([new_j_size])
    inc_top_tmp = np.zeros([new_i_size])
    inc_bottom_tmp = np.zeros([new_i_size])

    for i in xrange(0,new_i_size) :
      old_i = int(i)/int(self.n_div[0])
      inc_bottom_tmp[i] = self.inc_bottom[old_i]
      inc_top_tmp[i] = self.inc_top[old_i]
      for j in xrange(0,new_j_size) :
        old_j = int(j)/int(self.n_div[1]) 
        mat_id_tmp[i,j] = self.mat_id[old_i,old_j]
        src_id_tmp[i,j] = self.src_id[old_i,old_j]
    for j in xrange(0,new_j_size) :
      old_j = int(j)/int(new_j_size)
      inc_left_tmp[j] = self.inc_left[old_j]
      inc_right_tmp[j] = self.inc_right[old_j]

    self.mat_id = mat_id_tmp
    self.src_id = src_id_tmp
    self.inc_left = inc_left_tmp
    self.inc_right = inc_right_tmp
    self.inc_top = inc_top_tmp
    self.inc_bottom = inc_bottom_tmp

#----------------------------------------------------------------------------#

  def fokker_planck_xs(self,size) :
    """Compute the Fokker-Planck cross sections."""

    for i_mat in xrange(0,self.alpha.shape[0]) :
      pos = 0
# Compute the effective L_max used by the angular multigrid
      level = self.level
      L_max_eff = 2.**level*self.L_max
      for i in xrange(0,self.L_max) :
        for j in xrange(0,i+1) :
          self.sig_s[pos,i_mat] = self.alpha[i_mat]/2.0*(L_max_eff*\
              (L_max_eff+1)-i*(i+1))
          pos += 1
      if self.level!=0 :
        for i in xrange(pos,size) :
          self.sig_s[i,i_mat] = self.alpha[i_mat]/2.0*(L_max_eff*\
              (L_max_eff+1)-self.L_max*(self.L_max+1))

#----------------------------------------------------------------------------#

  def transport_correction(self) :
    """Compute the transport correction for the cross sections."""

    position={'1':0,'2':1,'4':6,'6':15,'8':15,'12':28,'16':45}
    for i_mat in xrange(0,self.alpha.shape[0]) :
      if self.optimal == True :
        if self.multigrid==False :
          if self.preconditioner=='P1SA' :
            if self.sig_s[:,i_mat].shape[0]>=4 :
              correction = (self.sig_s[3,i_mat]+self.sig_s[-1,i_mat])/2.
            else :
              correction = 0.
          else :
            if self.sig_s[:,i_mat].shape[0]>=2 and self.projection=='scalar' :
              correction = (self.sig_s[1,i_mat]+self.sig_s[-1,i_mat])/2.
            elif self.sig_s[:,i_mat].shape[0]>=4 and self.projection=='current' :
              correction = (self.sig_s[3,i_mat]+self.sigs_s[-1,i_mat])/2.
            else :
              correction = 0.
        else :
          if self.sig_s[:,i_mat].shape[0]>=2 and self.projection=='scalar' :
            pos = position[str(self.sn)]
            if pos!=0 :
              correction = (self.sig_s[pos,i_mat]+self.sig_s[-1,i_mat])/2
            else :
              correction = 0.
          elif self.sig_s[:,i_mat].shape[0]>=4 and self.projection=='current' :
            pos = position[str(self.sn)]
            if pos!=0 :
              correction = (self.sig_s[pos,i_mat]+self.sig_s[-1,i_mat])/2
            else :
              correction = 0.
          else :
            correction = 0.

      else :
        correction = self.sig_s[-1,i_mat]
      
      self.sig_t[i_mat] -= correction
      self.sig_s[:,i_mat] -= correction
