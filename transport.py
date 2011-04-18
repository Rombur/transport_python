# Python code
# Author: Bruno Turcksin
# Date: 2011-03-31 11:33:53.120619
#----------------------------------------------------------------------------#

# Driver for the transport code.
import parameters
import transport_solver
import output

tol = 1e-4
max_it = 1000
# If galerkin is True, sn = L_max -> the value of sn is not read
galerkin = True
fokker_planck = True
TC = False
optimal = True
is_precond = True
# Multigrid works only with S_8 due to a bug in gmres
# If multigrid is True is_precond is not read
multigrid = True
level = 0
L_max = 8
sn = 8
filename = 'transport'

param = parameters.parameters(galerkin,fokker_planck,TC,optimal,is_precond,
    multigrid,level,L_max,sn)
solver = transport_solver.transport_solver(param,tol,max_it)
solver.solve()
out = output.output(filename,solver.flux_moments,solver.p1sa_flxm,param)
out.write_in_file()
