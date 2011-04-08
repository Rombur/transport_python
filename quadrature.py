# Python code
# Author: Bruno Turcksin
# Date: 2011-03-31 14:02:08.144998

#----------------------------------------------------------------------------#
## Class quadrature                                                         ##
#----------------------------------------------------------------------------#

"""Contain the quadrature"""
      
import numpy as np
import scipy.special.orthogonal
import scipy.linalg
import scipy.misc.common as sci

class quadrature  :
  """Build the quadrature (Gauss-Legendre-Chebyshev) and the Galerkin
  version of the quadrature. Create the M and D matrices."""

  def __init__(self,param) :
    self.sn = param.sn
    self.n_dir = self.sn*(self.sn+2)/2
    self.galerkin = param.galerkin
    self.L_max = param.L_max
    self.n_mom = param.n_mom

#----------------------------------------------------------------------------#

  def build_quadrature(self) :
    """Build the quadrature, i.e. M, D and omega (direction vector)."""

# Compute the Gauss-Legendre quadrature
    [self.polar_nodes,self.polar_weight] = scipy.special.orthogonal.p_roots(self.sn) 

# Compute the Chebyshev quadrature
    [self.azith_nodes,self.azith_weight] = self.chebyshev()

    self.cos_theta = np.zeros((self.sn/2,1))
    for i in xrange(0,self.sn/2) :
      self.cos_theta[i] = self.polar_nodes[self.sn/2+i]
    self.sin_theta = np.sqrt(1-self.cos_theta**2)

# Compute omega on one octant
    self.build_octant()

# Compute omega by deploying the octant 
    self.deploy_octant()

# Compute the spherical harmonics
    self.compute_harmonics()

# Compute D
    if self.galerkin == True :
      self.D = scipy.linalg.inv(self.M)
    else :
      self.D = np.dot(self.M.transpose(),np.diag(self.weight))

#----------------------------------------------------------------------------#

  def chebyshev(self) :
    """Build the Chebyshev quadrature in a quadrant."""

    size = 0
    for i in xrange(1,self.sn/2+1) :
      size += i
    nodes = np.zeros((size,1))
    weight = np.zeros((size))

    pos = 0
    for i in xrange(0,self.sn/2) :
      for j in xrange(0,self.sn/2-i) :
        nodes[pos] = (np.pi/2.)/(self.sn/2-i)*j+(np.pi/4.)/(self.sn/2-i)
        weight[pos] = np.pi/(2.*(self.sn/2-i))
        pos += 1

    return nodes,weight

#----------------------------------------------------------------------------#

  def build_octant(self) :
    """Build omega and weight for one octant."""
    
    self.omega = np.zeros((self.n_dir,3))
    self.weight = np.zeros((self.n_dir))

    pos = 0
    offset = 0
    for i in xrange(0,self.sn/2) :
      for j in xrange(0,self.sn/2-i) :
        self.omega[pos,0] = self.sin_theta[i]*np.cos(self.azith_nodes[j+offset])
        self.omega[pos,1] = self.sin_theta[i]*np.sin(self.azith_nodes[j+offset])
        self.omega[pos,2] = self.cos_theta[i]
        self.weight[pos] = self.polar_weight[self.sn/2+i]*\
            self.azith_weight[j+offset]
        pos += 1
      offset += self.sn/2-i  

#----------------------------------------------------------------------------#

  def deploy_octant(self) :
    """Compute omega and the weights by deploing the octants."""

    n_dir_oct = self.n_dir/4
    offset = 0
    for i_octant in xrange(0,4) :
      if i_octant != 0 :
        for i in xrange (0,n_dir_oct) :
# Copy omega and weight 
          self.weight[i+offset] = self.weight[i]
          self.omega[i+offset,2] = self.omega[i,2]
# Correct omega signs
          if i_octant == 1 :
            self.omega[i+offset,0] = self.omega[i,0]
            self.omega[i+offset,1] = -self.omega[i,1]
          elif i_octant == 2 :
            self.omega[i+offset,0] = -self.omega[i,0]
            self.omega[i+offset,1] = -self.omega[i,1]
          else :
            self.omega[i+offset,0] = -self.omega[i,0]
            self.omega[i+offset,1] = self.omega[i,1]
      offset += n_dir_oct

    sum_weight = 0.
    for i in xrange(0,n_dir_oct) :
      sum_weight += 4 * self.weight[i]
    self.weight[:] = self.weight[:]/sum_weight

#----------------------------------------------------------------------------#

  def compute_harmonics(self) :
    """Compute the spherical harmonics and build the matrix M."""

    Ye = np.zeros((self.L_max+1,self.L_max+1,self.n_dir))
    Yo = np.zeros((self.L_max+1,self.L_max+1,self.n_dir))

    phi = np.zeros((self.n_dir,1))
    for i in xrange(0,self.n_dir) :
      phi[i] = np.arctan(self.omega[i,1]/self.omega[i,0])
      if self.omega[i,0] < 0. :
        phi[i] = phi[i] + np.pi

    for l in xrange(0,self.L_max+1) :
      for m in xrange(0,l+1) :
        P_ml =  scipy.special.lpmv(m,l,self.omega[:,2])
# Normalization of the associated Legendre polynomials
        if m == 0 :
          norm_P = P_ml
        else :
          norm_P = (-1.0)**m*np.sqrt(2*sci.factorial(l-m)/sci.factorial(l+m))\
              *P_ml
        size = norm_P.shape
        for i in xrange(0,size[0]) :
          Ye[l,m,i] = norm_P[i]*np.cos(m*phi[i])
          Yo[l,m,i] = norm_P[i]*np.sin(m*phi[i])

# Build the matrix M 
    self.sphr = np.zeros((self.n_dir,self.n_mom))
    self.M = np.zeros((self.n_dir,self.n_mom))
    if self.galerkin == True :
      for i in xrange(0,self.n_dir) :
        pos = 0
        for l in xrange(0,self.L_max+1) :
          fact = 2*l+1
          for m in xrange(l,-1,-1) :
# do not use the EVEN when m+l is odd for L<sn of L=sn and m=0
            if l<self.sn and np.fmod(m+l,2)==0 :
              self.sphr[i,pos] = Ye[l,m,i]
              self.M[i,pos] = fact*self.sphr[i,pos]
              pos += 1
          for m in xrange(1,l+1) :
# do not ise the ODD when m+l is odd for l<=sn
            if l<=self.sn and  np.fmod(m+l,2)==0 :
              self.sphr[i,pos] = Yo[l,m,i]
              self.M[i,pos] = fact*self.sphr[i,pos]
              pos += 1
    else :
      for i in xrange(0,self.n_dir) :
        pos = 0
        for l in xrange(0,self.L_max+1) :
          fact = 2*l+1
          for m in xrange(l,-1,-1) :
# do not use the EVEN when m+l is odd 
            if np.fmod(m+l,2)==0 :
              self.sphr[i,pos] = Ye[l,m,i]
              self.M[i,pos] = fact*self.sphr[i,pos]
              pos += 1
          for m in xrange(1,l+1) :
# do not ise the ODD when m+l is odd 
            if np.fmod(m+l,2)==0 :
              self.sphr[i,pos] = Yo[l,m,i]
              self.M[i,pos] = fact*self.sphr[i,pos]
              pos += 1
