#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

typedef vector<unsigned int> ui_vector;
typedef vector<int> i_vector;
typedef vector<double> d_vector;

/// Matrix-vector multiplication
d_vector mv_multiplication(d_vector const &values, ui_vector const &lines, 
    ui_vector const &columns, d_vector const &b);

/// Substraction of two vectors
d_vector vv_substraction(d_vector const &a, d_vector const &b, double alpha=1.);

/// Addition of two vectors
d_vector vv_addition(d_vector const &a, d_vector const &b, double alpha=1.);

/// Dot product between two vectors
double dot_product(d_vector const &a, d_vector const &b);

/// Conjugate gradient algorithm 
void cg(double* val, int n_val, int* lines, int n_lines, int* cols, 
    int n_cols, double* rhs, int n_rhs, double* solution, int n_solution, 
    double tol);

/// Extract the diagonal of the matrix
void extract_diagonal(d_vector const &values, ui_vector const &lines, 
    ui_vector const &columns, d_vector &diag);

/// Extract the lower triangular part of the matrix
void extract_lower_triangular(d_vector const &values, ui_vector const &lines,
    ui_vector const &columns, d_vector &lower_values, ui_vector &row_ptr,
    ui_vector &col_ind);

/// Forward substitution to solve a lower triangular system
d_vector forward_substitution(d_vector const &lower_values, ui_vector const 
    &row_ptr, ui_vector const &col_ind, d_vector const &diag, d_vector const &b);

/// Backward substitution to solve the transpose of a lower triangular system
d_vector backward_substitution(d_vector const &lower_values, ui_vector const 
    &row_ptr, ui_vector const &col_ind, d_vector const &diag, d_vector const &b);

/// Make the multiplication Ap for pcg
d_vector compute_Ap(d_vector const &lower_values, ui_vector const &row_ptr,
    ui_vector const &col_ind, d_vector const &diag, d_vector const &p);

/// Conjugate gradient algorithm preconditioned with SSOR
void pcg(double* val, int n_val, int* lines, int n_lines, int* cols, 
    int n_cols, double* rhs, int n_rhs, double* solution, int n_solution, 
    double tol, int* iter, int n_iter);

void d_print(d_vector const &a);

void ui_print(ui_vector const &a);
