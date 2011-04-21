# Python code
# Author: Bruno Turcksin
# Date: 2011-03-31 11:33:53.120619
#----------------------------------------------------------------------------#

# Driver for the transport code.
import parameters
import transport_solver
import output

# Tolerance for GMRES
tol = 1e-4
# Maximum number of iterations fo GMRES
max_it = 1000
# If galerkin is True, sn = L_max -> the value of sn is not read
galerkin = False
# If True uses Fokker-Planck cross-section
fokker_planck = False
# If True uses transport correction
TC = False
# If True and TC is True, uses the optimal transport correction. If TC is
# False, optimal is not read
optimal = True
# Preconditioner used : 'None', 'P1SA' or 'MIP'
preconditioner = 'MIP'
# Multigrid works only with S_8 due to a bug in gmres
multigrid = False
# L_max
L_max = 0
# Order of the Sn method
sn = 4
# Name of the output file
filename = 'transport'

# Driver of the program
param = parameters.parameters(galerkin,fokker_planck,TC,optimal,preconditioner,
    multigrid,L_max,sn)
solver = transport_solver.transport_solver(param,tol,max_it)
solver.solve()
out = output.output(filename,solver.flux_moments,solver.p1sa_flxm,
    solver.mip_flxm,param)
out.write_in_file()
