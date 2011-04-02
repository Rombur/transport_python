# Python code
# Author: Bruno Turcksin
# Date: 2011-03-31 11:33:53.120619
#----------------------------------------------------------------------------#

# Driver for the transport code.

import parameters
import transport_solver

tol = 1e-12
max_it = 1000
param = parameters.parameters(True,True,True,True)
#param = parameters.parameters(False,False,False,False)
solver = transport_solver.transport_solver(param,tol,max_it)
solver.solve()
