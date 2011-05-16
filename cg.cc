#include "cg.hh"
#include <iostream>

d_vector mv_multiplication(d_vector const &values, ui_vector const &lines, 
    ui_vector const &columns, d_vector const &b)
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
   * Conjugate gradient algorithm. The matrix is given using 3 arrays : the 
   * values, the indices of the lines and the indices of the columns. This is 
   * not a csr format because I don't know how to get the 3 arrays from python. 
   * The initial guess is zero.
   */

  const unsigned int size(n_rhs);

  // Convert the arrays to vector because I hate arrays.
  d_vector values(a_values,a_values+n_values);
  ui_vector lines(a_lines,a_lines+n_lines);
  ui_vector columns(a_columns,a_columns+n_columns);
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
    
    cout<<"iteration "<<i<<", "<<rs_new<<endl;
    if (sqrt(rs_new)<tol)
    {
      cout<<"converged"<<endl;
      break;
    }

    p = vv_addition(r,p,rs_new/rs_old);
    rs_old = rs_new;
  }

  // Copy the solution in the array solution
  for (unsigned int i=0; i<size;i++)
    a_solution[i] = solution[i];
}

void extract_diagonal(d_vector const &values, ui_vector const &lines, 
    ui_vector const &columns, d_vector &diag)
{
  /**
   * Extract the diagonal of the matrix.
   */

  const unsigned int size(lines.size());
  unsigned int pos(0);

  for (unsigned int i=0; i<size; ++i)
  {
    if (lines[i]==columns[i])
    {
      diag[pos] = values[i];
      ++pos;
    }
  }
}

void extract_lower_triangular(d_vector const &values, ui_vector const &lines,
    ui_vector const &columns, d_vector &lower_values, ui_vector &row_ptr,
    ui_vector &col_ind)
{
  /***
   * Extract the lower triangular part of the matrix and store it to the CSR
   * format.
   */

  const unsigned int size(values.size());
  unsigned int last_line(0);

  for (unsigned int i=0; i<size; ++i)
  {

    // We cannot skip a line when using the CSR format. If the matrix contains
    // only diagonal elements, we add a 0.
    if (i!=0 && columns[i]==lines[i])
    {
      if (lines[i-1]!=lines[i])
      {
        lower_values.push_back(0.);
        col_ind.push_back(columns[i-1]);
        row_ptr.push_back(col_ind.size()-1);
        last_line = lines[i];
      }
    }
    if (columns[i]<lines[i])
    {
      lower_values.push_back(values[i]);
      col_ind.push_back(columns[i]);
      if (last_line!=lines[i])
      {
        row_ptr.push_back(col_ind.size()-1);
        last_line = lines[i];
      }
    }
  }
}   

d_vector forward_substitution(d_vector const &lower_values, ui_vector const 
    &row_ptr, ui_vector const &col_ind, d_vector const &diag, d_vector const &b)
{
  /**
   * Multiply the inverse of a lower triangular by a vector. It is the same
   * to solve the system Lx=b than to multipy inv(L) by b.
   */

  const unsigned int size(b.size());
  d_vector result(size,0.0);

  result[0] = b[0]/diag[0];

  for (unsigned int i=0; i<size-1; ++i)
  {
    unsigned int end_line;
    if (i==size-2)
      end_line = lower_values.size();
    else
      end_line = row_ptr[i+1];

    result[i+1] = b[i+1];

    for (unsigned int j=row_ptr[i]; j<end_line; ++j)
      result[i+1] -= lower_values[j]*result[col_ind[j]];

    result[i+1] /= diag[i+1];
  }

  return result;
}

d_vector backward_substitution(d_vector const &lower_values, ui_vector const 
    &row_ptr, ui_vector const &col_ind, d_vector const &diag, d_vector const &b)
{
  /**
   * Multiply the inverse of the transpose of lower triangular by a vector.
   * It is the same to solve the system L^Tx=b than to multipy inv(L^T) by b.
   */

  const unsigned int size(b.size());
  const unsigned int lv_size(lower_values.size());
  const unsigned int row_size(row_ptr.size());
  d_vector result(size,0.0);

  for (int i=size-1; i>=0; --i)
  {
    result[i] = b[i];
    for (int j=lv_size-1; j>=0; --j)
      if (int(col_ind[j])==i)
      {
        unsigned int pos(0);
        for (unsigned int k=0; k<row_size; ++k)
          if (int(row_ptr[k]) <= j)
            pos = k+1;
          else
            break;
        result[i] -= lower_values[j]*result[pos];
      }
    result[i] /= diag[i];
  }

  return result;
}

d_vector compute_Ap(d_vector const &lower_values, ui_vector const &row_ptr,
    ui_vector const &col_ind, d_vector const &diag, d_vector const &p)
{
  /**
   * Make the multiplication Ap but when A is modified by the preconditioner.
   */

  const unsigned int size(diag.size());
  d_vector t,result,tmp(diag.size());

  t = backward_substitution(lower_values,row_ptr,col_ind,diag,p);

  for (unsigned int i=0; i<size; ++i)
    tmp[i] = diag[i]*t[i];

  tmp = vv_substraction(p,tmp);
  tmp = forward_substitution(lower_values,row_ptr,col_ind,diag,tmp);
  result = vv_addition(t,tmp);

  return result;
}

void pcg(double* a_values, int n_values, int* a_lines, int n_lines, 
    int* a_columns, int n_columns, double* a_rhs, int n_rhs, double* a_solution, 
    int n_solution, double tol, int* iter, int n_iter)
{
  /**
   * Conjugate gradient algorithm preconditioned with SSOR using Eisenstat's
   * trick. The matrix is given using 3 arrays : the values, the indices of the 
   * lines and the indices of the columns. This is not a csr format because 
   * I don't know how to get the 3 arrays from python. The initial guess is zero.
   */

  const unsigned int size(n_rhs);
  double r0_rp0;

  // Convert the arrays to vector because I hate arrays.
  d_vector values(a_values,a_values+n_values);
  d_vector rhs(a_rhs,a_rhs+n_rhs);
  ui_vector lines(a_lines,a_lines+n_lines);
  ui_vector columns(a_columns,a_columns+n_columns);
  d_vector solution(n_solution,0.);

  d_vector r(size,0.0);
  d_vector p(size),r_prime(size),diag(size);
  d_vector lower_values;
  ui_vector col_ind,row_ptr;

  // Extract the diagonal of the matrix
  extract_diagonal(values,lines,columns,diag);
  
  // Extract the lower triangular part of the matrix
  extract_lower_triangular(values,lines,columns,lower_values,row_ptr,col_ind); 

  r = forward_substitution(lower_values,row_ptr,col_ind,diag,rhs);

  for (unsigned int i=0; i<size; ++i)
    r_prime[i] = diag[i]*r[i];
  p = r_prime;

  r0_rp0 = dot_product(r,r_prime);

  // If the rhs is zero the solution is zeros
  bool r_is_zero(true);
  for (unsigned int i=0; i<size && r_is_zero; ++i)
    if (r[i]<-1e-20 && r[i]>1e-20)
      r_is_zero = false;

  if (r_is_zero==false)
  {
    for (unsigned int i=0; i<size; ++i)
    {
      double alpha,beta,num,denom,r_rp;
      d_vector Ap,tmp;

      Ap = compute_Ap(lower_values,row_ptr,col_ind,diag,p);

      r_rp = dot_product(r,r_prime);
      denom = dot_product(p,Ap);
      alpha = r_rp/denom;

      tmp = backward_substitution(lower_values,row_ptr,col_ind,diag,p);

      solution = vv_addition(solution,tmp,alpha);

      r = vv_substraction(r,Ap,alpha);
      for (unsigned int j=0; j<size; ++j)
        r_prime[j] = diag[j]*r[j];
      num = dot_product(r,r_prime);
      beta = num/r_rp;

      iter[0] = i;
      if (num/r0_rp0<tol)
      {
        cout<<"converged "<<num/r0_rp0<<" "<<i<<endl;
        break;
      }

      p = vv_addition(r_prime,p,beta);
    }
  }
  
  // Copy the solution in the array solution
  for (unsigned int i=0; i<size;i++)
    a_solution[i] = solution[i];
}

void d_print(d_vector const &a)
{
  for (unsigned int i=0; i<a.size(); i++)
    cout<<a[i]<<" ";
  cout<<endl;
}

void ui_print(ui_vector const &a)
{
  for (unsigned int i=0; i<a.size(); i++)
    cout<<a[i]<<" ";
  cout<<endl;
}
