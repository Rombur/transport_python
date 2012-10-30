# Python code
# Author: Bruno Turcksin
# Date: 2011-03-31 11:33:53.120619
#----------------------------------------------------------------------------#

# Driver for the transport code.

import subprocess
import numpy as np
import PARAMETERS
import FE_TRANSPORT_SOLVER
import SV_TRANSPORT_SOLVER
import OUTPUT

# Tolerance for GMRES
tol = 1e-4
# Maximum number of iterations fo GMRES
max_it = 10000
# If galerkin is True, sn = L_max -> the value of sn is not read
galerkin = False
# L_max
L_max = 0
# Order of the Sn method
sn = 8
# Name of the output file
filename = 'transport'
# Spatial discretization ('FE' or 'SV')
discretization = 'SV'
# If SV is used, there are two different_discretization possible
point_value = False
# If print_to_file is True, the message are written on a file, otherwise they
# are printed on the scree
print_to_file = False
# Toggle between SI and GMRES when multigrid is not use
gmres = True

x_width = np.array([1.,1.])
y_width = np.array([1.,1.])
n_x_div = np.array([2,4])
n_y_div = np.array([4,2])
inc_bottom = np.array([0.,0.])
inc_right = np.array([0.,0.])
inc_top = np.array([0.,0.])
inc_left = np.array([0.,0.])
mat_id = np.array([[0,1],[0,1]])
src_id = np.array([[0,0],[1,1]])
src = np.array([10.,0.])
sig_t = np.array([2.,1.])
sig_s = np.zeros((2,1))
sig_s[0,0] = 1.
sig_s[1,0] = 0.
weight = 2.*np.pi
verbose = 1

# Driver of the program
try :
  output_file = open(filename+'.txt','w')
  param = PARAMETERS.PARAMETERS(galerkin,L_max,sn,print_to_file,gmres,x_width,y_width,
      n_x_div,n_y_div,inc_bottom,inc_right,inc_top,inc_left,mat_id,src_id,src,
      sig_t,sig_s,weight,verbose,discretization)
  if discretization=='FE' :
    solver = FE_TRANSPORT_SOLVER.FE_TRANSPORT_SOLVER(param,tol,max_it,output_file)
  else :
    solver = SV_TRANSPORT_SOLVER.SV_TRANSPORT_SOLVER(param,tol,max_it,
        output_file,point_value)
  solver.Solve()
  out = OUTPUT.OUTPUT(filename,solver.flux_moments,param,solver.dof_handler.n_dof)
  out.Write_in_file()
  if print_to_file==False :
    subprocess.call('rm '+filename+'.txt',shell=True)
except NameError :
  raise
