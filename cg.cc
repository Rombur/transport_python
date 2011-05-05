#include "cg.hh"

d_vector mv_multiplication(d_vector const &values, i_vector const &lines, 
    i_vector const &columns, d_vector const &b)
{
  /**
   * Matrix-vector multiplication.
   */

  const unsigned int size(b.size());
  const unsigned int nnz(values.size());
  d_vector result(size,0.0);

  for (unsigned int i=0; i<nnz; ++i)
    result[lines[i]] += values[i]*b[columns[i]];

  return result;
}

d_vector vv_substraction(d_vector const &a, d_vector const &b, double alpha)
{
  /**
   * Substraction of the vector a by the vector alpha b. By default, alpha = 1. 
   */

  const unsigned int size(a.size());
  d_vector result(size);

  for (unsigned int i=0; i<size; ++i)
    result[i] = a[i]-alpha*b[i];

  return result;
}

d_vector vv_addition(d_vector const &a, d_vector const &b, double alpha)
{
  /**
   * Addition of the vectors a and alpha b. By default, alpha = 1.
   */

  const unsigned int size(a.size());
  d_vector result(size);

  for (unsigned int i=0; i<size; ++i)
    result[i] = a[i]+alpha*b[i];

  return result;
}

double dot_product(d_vector const &a, d_vector const &b)
{
  /**
   * Dot product between the vector a and the vector b. Return a double.
   */

  const unsigned int size(a.size());
  double result(0.);

  for (unsigned int i=0; i<size; ++i)
    result += a[i]*b[i];

  return result;
}

void cg(double* a_values, int n_values, int* a_lines, int n_lines, 
    int* a_columns, int n_columns, double* a_rhs, int n_rhs, double* a_solution, 
    int n_solution, double tol)
{
  /**
   * Conjugate gradient algorithm preconditioned with SSOR. The matrix is
   * given using 3 arrays : the values, the indices of the lines and the
   * indices of the columns. This is not a csr format because I don't know how
   * to get the 3 arrays from python. The initial guess is zero.
   */

  const unsigned int size(n_rhs);

  // Convert the arrays to vector because I hate arrays.
  d_vector values(a_values,a_values+n_values);
  i_vector lines(a_lines,a_lines+n_lines);
  i_vector columns(a_columns,a_columns+n_columns);
  d_vector solution(n_solution,0.);

  d_vector r(a_rhs,a_rhs+n_rhs);
  d_vector p(r);
  // Compute the norm of the residual
  double rs_old = dot_product(r,r);

  for (unsigned int i=0; i<size; ++i)
  {
    double alpha,denom,rs_new;
    d_vector Ap;
    
    Ap = mv_multiplication(values,lines,columns,p);
    denom = dot_product(p,Ap);
    alpha = rs_old/denom;
    solution = vv_addition(solution,p,alpha);
    r = vv_substraction(r,Ap,alpha);  
    rs_new = dot_product(r,r);
    
    if (sqrt(rs_new)<tol)
      break;

    p = vv_addition(r,p,rs_new/rs_old);
    rs_old = rs_new;
  }

  // Copy the solution in the array solution
  for (unsigned int i=0; i<size;i++)
    a_solution[i] = solution[i];
}
