# Python code
# Author: Bruno Turcksin
# Date: 2011-04-03 14:31:35.164159

#----------------------------------------------------------------------------#
## Class output                                                             ##
#----------------------------------------------------------------------------#

"""Create the output file"""

import numpy as np
import enthought.mayavi as mayavi
import parameters

class output(object) :
  """Write the solution in a file."""

  def __init__(self,filename,flux_moments,p1sa_flxm,mip_flxm,param) :

    super(output,self).__init__()
    self.filename = filename
    self.flux_moments = flux_moments
    self.p1sa_flxm= p1sa_flxm
    self.mip_flxm= mip_flxm
    self.param = param
    self.compute_grid()

#----------------------------------------------------------------------------#

  def compute_grid(self) :
    """Compute the 2D cartesian grid."""
    
    x_size = int(2*self.param.n_x)
    y_size = int(2*self.param.n_y)
    self.x = np.zeros(x_size)
    self.y = np.zeros(y_size)

    for i in xrange(1,x_size-1,2) :
      self.x[i] = self.x[i-1]+self.param.width_x_cell
      self.x[i+1] = self.x[i-1]+self.param.width_x_cell
    self.x[x_size-1] = self.x[x_size-2]+self.param.width_x_cell
    
    for i in xrange(1,y_size-1,2) :
      self.y[i] = self.y[i-1]+self.param.width_y_cell
      self.y[i+1] = self.y[i-1]+self.param.width_y_cell
    self.y[y_size-1] = self.y[y_size-2]+self.param.width_y_cell

#----------------------------------------------------------------------------#

  def write_in_file(self) :
    """Write the moment flux and the mesh in a file."""

# Write the mesh  and the flux
    np.savez(self.filename,x=self.x,y=self.y,
      flux_moments=self.flux_moments,p1sa_flxm=self.p1sa_flxm,
      mip_flxm=self.mip_flxm)
