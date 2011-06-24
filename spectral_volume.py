# Python code
# Author: Bruno Turcksin
# Date: 2011-06-10 15:09:38.369341

#----------------------------------------------------------------------------#
## Class spectral_volume                                                    ##
#----------------------------------------------------------------------------#

"""Contain the matrices needed for the spectral volumes"""

import numpy as np

class spectral_volume(object)  :
  """Contain all the matrices needed to build the spectral volumes."""

  def __init__(self) :

    super(spectral_volume,self).__init__()
    self.width = param.width
    self.width_cell = np.array([self.width[0]/param.n_div[0],self.width[1]/
      param.n_div[1]])
    self.delta = self.width_cell/2.

#----------------------------------------------------------------------------#

  def build_edge_integral(self,point_value) :
    """Build the matrices needed to perform the countour integral."""

    den = self.delta[0]*self.delta[1]
    den_0 = den
    den_1 = -den
    den_2 = -den
    den_3 = den
    
# First CV, bottom left (arrays are 1 line each)
    self.bottom_edge_cv_1 = -self.delta[0]**2/2.*self.delta[1]*\
        np.array([1.5/den_0,0.5/den_1.,1.5/den_2,0.5/den_3]) +\
        self.delta[0]*np.array([2.25,-0.75,-0.75,0.25])

    self.right_edge_cv_1 = self.delta[0]*self.delta[1]**2/2.*np.array(\
        [1./den_0,1./den_1,1./den_2,1./den_3]) - self.delta[0]*self.delta[1]*\
        np.array([1.5/den_0,0.5/den_1,1.5/den_2,0.5/den_3])-self.delta[1]**2/2.*\
        self.delta[0]*np.array([1.5/den_0,1.5/den_1,0.5/den_2,0.5/den_3])-\
        self.delta[1]*np.array([2.25,-0.75,-0.75,0.25])

    self.top_edge_cv_1 = -self.delta[0]**2/2.self.delta[0]*np.array(\
        [1./den_0,1./den_1,1./den_2,1./den_3])+self.delta[0]**2/2.*self.delta[1]*\
        np.array([1.5/den_0,0.5/den_1,1.5/den_2,0.5/den_3])-self.delta[1]*\
        self.delta[0]*np.array(1.5/den_0,1.5/den_1,0.5/den_2,0.5/den_3)-\
        self.delta[0]*np.array([2.25,-0.75,-0.75,0.25])

    self.left_edge_cv_1 = self.delta[1]**2/2.*self.delta[0]*np.array(\
        [1.5/den_0,1.5/den_1,0.5/den_2,0.5/den_3])-self.delta[1]*np.array(\
        [2.25,-0.75,-0.75,0.25])

    if point_value==True :
      self.surface_cv_1 = np.array([1.,0.,0.,0.])
    else :
      self.surface_cv_1 = (self.delta[0]*self.delta[1])**2/4.*np.array(\
          [1./den_0,1./den_1,1./den_2,1./den_3])-(self.delta[0]*self.delta[1])**2/\
          2.*np.array([1.5,0.5,1.5,0.5])-(self.delta[0]*self.delta[1])**2/2.*\
          np.array([1.5/den_0,1.5/den_1,0.5/den_2,0.5/den_3])+self.delta[0]*\
          self.delta[1]*np.array([2.25,-0.75,-0.75,0.25])

# Second CV, bottom right
    self.bottom_edge_cv_2 = -self.delta[0]**2/2.*self.delta[1]*np.array(\
        [1.5/den_0,0.5/den_1,1.5/den_2,0.5/den_3])+self.delta[0]*np.array(\
        [2.25,-0.75,-0.75,0.25])
    
    self.right_edge_cv_2 = self.delta_x[0]*self.delta[1]**2*np.array(\
        [1./den_0,1./den_1,1./den_2,1./den_3])-2.*self.delta[0]*self.delta[1]*\
        np.array([1.5/den_0,0.5/den_1,1.5/den_2,0.5/den_3])-self.delta[1]**2/2.*\
        self.delta[0]*np.array([1.5/den_0,1.5/den_1,0.5/den_2,0.5/den_3])+\
        self.delta[1]*np.array([2.25,-0.75,-0.75,0.25])

    self.top_edge_cv_2 = -self.delta[0]**2/2*self.delta[1]*np.array(\
        [1./den_0,1./den_1,1./den_2,1./den_3]) + self.delta[0]**2*\
        self.delta[1]*np.array([1.5/den_0,0.5/den_1,1.5/den_2,0.5/den_3])-\
        self.delta[0]*self.delta[1]*np.array([1.5/den_0,1.5/den_1,0.5/den_2,\
        0.5/den_3])

    self.left_edge_cv_2 = -self.delta[0]*self.delta[1]**2/2.*np.array(\
        [1./den_0,1./den_1,1./den_2,1./den_3])-self.delta[0]*self.delta[1]*\
        np.array([1.5/den_0,0.5/den_1,1.5/den_2,0.5/den_3])+self.delta[1]**2*\
        self.delta[0]/2.*np.array([1.5/den_0,1.5/den_1,0.5/den_2,0.5/den_3])-\
        self.delta[1]*np.array([2.25,-0.75,-0.75,0.25])
  
    if point_value==True :
      self.surface_cv_2 = np.array([0.,1.,0.,0.])
    else :
      self.surface_cv_2 = (self.delta[0]*self.delta[1])**2/4.*np.array(\
          [1./den_0,1/den_1.,1./den_2,1./den_3])-(self.delta[0]*self.delta[1])**2/\
          2*np.array([1.5,0.5,1.5,0.5])-(self.delta[0]*self.delta[1])**2/2.*\
          np.array([1.5/den_0,1.5/den_1,0.5/den_2,0.5/den_3]) + self.delta[0]*\
          self.delta[1]*np.array([2.25,-0.75,-0.75,0.25])

# Third CV, top right        
    self.bottom_edge_cv_1 = self.delta[0]**2/2.*self.delta[1]*np.array(\
        [1./den_0,1./den_1,1./den_2,1./den_3])-self.delta[0]**2/2.*\
        self.delta[1]*np.array([1.5/den_0,0.5/den_1,1.5/den_2,0.5/den_3])-\
        self.delta[0]*self.delta[1]*np.array(\
        [1.5/den_0,1.5/den_1,0.5/den_2,0.5/den_3])

#----------------------------------------------------------------------------#

  def build_cell_integral(self,efsf)