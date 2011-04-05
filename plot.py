# Python code
# Author: Bruno Turcksin
# Date: 2011-04-03 23:29:00.078307

#----------------------------------------------------------------------------#
## Module plot                                                              ##
#----------------------------------------------------------------------------#

"""Module which loads the mesh and the flux and then create the mayavi 
objects."""

import numpy as np
import enthought.mayavi.tools.pipeline as mayavi_pipeline
import enthought.mayavi.mlab as mayavi_mlab

def create_mayavi(filename) :
  """Open mayavi."""

# Load mesh and solution
  data = np.load(filename+'.npz')
  x = data['x']
  x_size = x.shape[0]
  y = data['y']
  y_size = y.shape[0]
# Load flux moments
  flux_moments = data['flux_moments']

  scalar_flux = flux_moments[0:4*x_size*y_size]
  new_scalar_flux = change_shape(x_size,y_size,scalar_flux)
  new_x,new_y = expand(x,y,x_size,y_size)
  mayavi_mlab.mesh(new_x,new_y,new_scalar_flux)
#  mayavi_pipeline.array2d_source(x,y,new_scalar_flux,name='2D')
#  mayavi_mlab.surf(x,y,new_scalar_flux,name='3D')

#----------------------------------------------------------------------------#

def expand(x,y,x_size,y_size) :
  """Create 2D array of x and y."""

  new_x = np.zeros([x_size,y_size])
  new_y = np.zeros([x_size,y_size])

  for i in xrange(0,x_size) :
    for j in xrange(0,y_size) :
      new_x[i,j] = x[i]

  for i in xrange(0,x_size) :
    for j in xrange(0,y_size) :
      new_y[i,j] = y[j]

  return new_x,new_y

#----------------------------------------------------------------------------#

def change_shape(x_size,y_size,flux) :
  """Change the size of the flux."""

  new_flux = np.zeros([x_size,y_size])
  for i in xrange(0,x_size) :
    for j in xrange(0,y_size) :
      pos = mapping(i,j,x_size)
      new_flux[i,j] = flux[pos]

  return new_flux

#----------------------------------------------------------------------------#

def  mapping(i,j,x_size) :
  """Compute the position in the flux_moments vector."""

  return int(2*i+np.fmod(j,2)+np.floor(j/2)*2*x_size)
