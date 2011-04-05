# Python code
# Author: Bruno Turcksin
# Date: 2011-03-31 11:33:53.120619
#----------------------------------------------------------------------------#

# Driver for the transport code.

import parameters
import transport_solver
import output

tol = 1e-6
max_it = 1000
galerkin = False
fokker_planck = False
TC = False
optimal = False
filename = 'transport'

param = parameters.parameters(galerkin,fokker_planck,TC,optimal)
#param = parameters.parameters(False,False,False,False)
solver = transport_solver.transport_solver(param,tol,max_it)
solver.solve()
out = output.output(filename,solver.flux_moments,param)
out.write_in_file()
