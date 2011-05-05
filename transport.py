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
galerkin = True
# If True uses Fokker-Planck cross-section
fokker_planck = True
# If True uses transport correction
TC = True
# If True and TC is True, uses the optimal transport correction. If TC is
# False, optimal is not read
optimal = True
# Preconditioner used : 'None', 'P1SA' or 'MIP'
preconditioner = 'MIP'
# When multigrid is used, a preconditioner has to be applied (P1SA is chosen by
# default)
# Multigrid works only when using with Fokker-Planck cross-section
multigrid = True
# L_max
L_max = 8
# Order of the Sn method
sn = 8
# Name of the output file
filename = 'transport'

# Driver of the program
output_file = open(filename+'.txt','w')
param = parameters.parameters(galerkin,fokker_planck,TC,optimal,preconditioner,
    multigrid,L_max,sn)
solver = transport_solver.transport_solver(param,tol,max_it,output_file)
solver.solve()
out = output.output(filename,solver.flux_moments,solver.p1sa_flxm,
    solver.mip_flxm,param)
out.write_in_file()
