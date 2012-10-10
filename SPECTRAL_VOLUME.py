# Python code
# Author: Bruno Turcksin
# Date: 2011-06-10 15:09:38.369341

#----------------------------------------------------------------------------#
## Class SPECTRAL_VOLUME                                                    ##
#----------------------------------------------------------------------------#

"""Contain the matrices needed for the spectral volumes"""

import numpy as np

class SPECTRAL_VOLUME(object)  :
  """Contain all the matrices needed to build the spectral volumes."""

  def __init__(self,width) :

    super(SPECTRAL_VOLUME,self).__init__()
    self.width = width
#    self.delta = self.width/2.
    self.delta = 1.
    self.n_dof_per_cell = 4

#----------------------------------------------------------------------------#

  def Build_edge_integral(self) :
    """Build the matrices needed to perform the countour integral."""

    den = self.delta[0]*self.delta[1]
    den_0 = den
    den_1 = -den
    den_2 = -den
    den_3 = den 

    array_0 = np.array([1./den_0,1./den_1,1./den_2,1./den_3])
    array_1 = self.delta[1]*np.array([1.5/den_0,0.5/den_1,1.5/den_2,0.5/den_3])
    array_2 = self.delta[0]*np.array([1.5/den_0,1.5/den_1,0.5/den_1,0.5/den_3])
    array_3 = np.array([2.25,-0.75,-0.75,0.25])
    
# First CV, bottom left (arrays are 1 line each)
    self.bottom_edge_cv_0 = -self.delta[0]**2/2.*array_1+self.delta[0]*array_3

    self.right_edge_cv_0 = self.delta[0]*self.delta[1]**2/2.*array_0-\
        self.delta[0]*array_1-self.delta[1]**2/2.*array_2-self.delta[1]*array_3

    self.top_edge_cv_0 = -self.delta[0]**2/2.*self.delta[0]*array_0+\
        self.delta[0]**2/2.*array_1-self.delta[1]*array_2-self.delta[0]*array_3

    self.left_edge_cv_0 = self.delta[1]**2/2.*self.delta[0]*array_2-\
        self.delta[1]*array_3

# Second CV, bottom right
    self.bottom_edge_cv_1 = -self.delta[0]**2/2.*array_1+self.delta[0]*array_3
    
    self.right_edge_cv_1 = 2.*self.delta[0]*self.delta[1]**2*array_0-\
        2.*self.delta[0]*array_1-self.delta[1]**2/2.*array_3+self.delta[1]*array_3

    self.top_edge_cv_1 = -self.delta[0]**2/2*self.delta[1]*array_0+\
        self.delta[0]**2*array_1-self.delta[1]*array_2-self.delta[0]*array_3

    self.left_edge_cv_1 = -self.delta[0]*self.delta[1]**2/2.*array_0-\
        self.delta[0]*array_1+self.delta[1]**2/2.*array_2-self.delta[1]*array_3

# Third CV, top right        
    self.bottom_edge_cv_2= self.delta[0]**2/2.*self.delta[1]*array_0-\
        self.delta[0]**2/2.*array_1-self.delta[1]*array_2+self.delta[0]*array_3

    self.right_edge_cv_2 = 2.*self.delta[0]*self.delta[1]**2/2.*array_1-\
        2*self.delta[0]*array_1-self.delta[1]**2/2.*array_2+self.delta[1]*array_3

    self.top_edge_cv_2 = -self.delta[0]**2/2.*2.*self.delta[1]*array_0+\
        self.delta[0]**2/2.*array_1-2.*self.delta[1]*array_2-self.delta[0]*\
        array_3
    
    self.left_edge_cv_2 = -self.delta[0]*self.delta[1]**2/2.*array_0-\
        self.delta[0]*array_1+self.delta[1]**2/2.*array_2-self.delta[1]*array_3

# Fourth CV, top left 
    self.bottom_edge_cv_3 = self.delta[0]**2/1.*self.delta[1]*array_0-\
        self.delta[0]**2/2.*array_1-self.delta[1]*array_2+self.delta[0]*array_3

    self.right_edge_cv_3 = self.delta[0]*self.delta[1]**2/2.*array_0-\
        self.delta[0]*array_1-self.delta[0]**2/2.*array_2+self.delta[1]*array_3

    self.top_edge_cv_3 = -self.delta[0]**2/2.*2.*self.delta[1]*array_0+\
        self.delta[0]**2/2.*array_1-2.*self.delta[1]*array_2-self.delta[0]*array_3

    self.left_edge_cv_3 = self.delta[1]**2/2.*array_2-self.delta[1]*array_3    

#----------------------------------------------------------------------------#

  def Build_cell_integral(self,point_value) :
    """Build the matrices needed to perform the surface integral. point_value
    is a boolean used to know how to build the array."""

    if point_value==True :
      self.surface_cv_0 = np.array([1.,0.,0.,0.])
      self.surface_cv_1 = np.array([0.,1.,0.,0.])
      self.surface_cv_2 = np.array([0.,0.,1.,0.])
      self.surface_cv_3 = np.array([0.,0.,0.,1.])
    else :
      den = self.delta[0]*self.delta[1]
      den_0 = den
      den_1 = -den
      den_2 = -den
      den_3 = den 

      array_0 = np.array([1./den_0,1./den_1,1./den_2,1./den_3])
      array_1 = self.delta[1]*np.array([1.5/den_0,0.5/den_1,1.5/den_2,0.5/den_3])
      array_2 = self.delta[0]*np.array([1.5/den_0,1.5/den_1,0.5/den_1,0.5/den_3])
      array_3 = np.array([2.25,-0.75,-0.75,0.25])

      self.surface_cv_0 = (self.delta[0]*self.delta[1])**2/4.*array_0-\
          (self.delta[0])**2/2.*self.delta[1]*array_1-self.delta[0]*\
          self.delta[1]**2/2.*array_2+self.delta[0]*self.delta[1]*array_3
      self.surface_cv_1 = (self.delta[0]*self.delta[1])**2/4.*array_0-\
          self.delta[0]**2*self.delta[1]/2.*array_1-self.delta[0]*\
          self.delta[1]**2/2.*array_2+self.delta[0]*self.delta[1]*array_3
      self.surface_cv_2 = (self.delta[0]*self.delta[1])**2/4.*array_0-\
          self.delta[0]**2/2.*self.delta[1]*array_1-self.delta[0]*\
          self.delta[1]**2/2.*array_2+self.delta[0]*self.delta[1]*array_3
      self.surface_cv_3 = (self.delta[0]*self.delta[1])**2/4.*array_0-\
          self.delta[0]**2/2.*self.delta[1]*array_1-self.delta[0]*\
          self.delta[1]**2/2.*array_3+self.delta[0]*self.delta[1]*array_3
    
    self.surface_sv = np.array([self.surface_cv_0,self.surface_cv_1,\
        self.surface_cv_2,self.surface_cv_3])
