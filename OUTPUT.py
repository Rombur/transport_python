# Python code
# Author: Bruno Turcksin
# Date: 2011-04-03 14:31:35.164159

#----------------------------------------------------------------------------#
## Class OUTPUT                                                             ##
#----------------------------------------------------------------------------#

"""Create the output file"""

import numpy as np
import PARAMETERS

class OUTPUT(object) :
  """Write the solution in a file."""

  def __init__(self,filename,flux_moments,param,n_dof) :

    super(OUTPUT,self).__init__()
    self.filename = filename
    self.flux_moments = flux_moments
    self.param = param
    self.Compute_grid()
    self.Compute_scalar_flux(n_dof)

#----------------------------------------------------------------------------#

  def Compute_grid(self) :
    """Compute the 2D cartesian grid."""
    
    x_size = int(2*self.param.n_x)
    y_size = int(2*self.param.n_y)
    self.x = np.zeros(x_size)
    self.y = np.zeros(y_size)

    for i in range(1,x_size-1,2) :
      self.x[i] = self.x[i-1]+self.param.x_width[i%2]
      self.x[i+1] = self.x[i-1]+self.param.x_width[i%2]
    self.x[x_size-1] = self.x[x_size-2]+self.param.x_width[(x_size-1)%2]
    
    for i in range(1,y_size-1,2) :
      self.y[i] = self.y[i-1]+self.param.y_width[i%2]
      self.y[i+1] = self.y[i-1]+self.param.y_width[i%2]
    self.y[y_size-1] = self.y[y_size-2]+self.param.y_width[(y_size-1)%2]

#----------------------------------------------------------------------------#

  def Compute_scalar_flux(self,n_dof) :
    """Compute the scalar flux given the moments of the flux and the weight."""

    self.scalar_flux = np.zeros(n_dof)
    weight = np.sqrt(self.param.weight)
    for i in range(n_dof) :
      self.scalar_flux[i] = weight*self.flux_moments[i]

#----------------------------------------------------------------------------#

  def Write_in_file(self) :
    """Write the moment flux and the mesh in a file."""

# Write the mesh and the flux
    np.savez(self.filename,x=self.x,y=self.y,flux_moments=self.flux_moments,
        scalar_flux=self.scalar_flux)
