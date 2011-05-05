#include <vector>
#include <cmath>

using namespace std;

typedef vector<double> d_vector;
typedef vector<int> i_vector;

/// Matrix-vector multiplication
d_vector mv_multiplication(d_vector const &values, i_vector const &lines, 
    i_vector const &columns, d_vector const &b);

/// Substraction of two vectors
d_vector vv_substraction(d_vector const &a, d_vector const &b, double alpha=1.);

/// Addition of two vectors
d_vector vv_addition(d_vector const &a, d_vector const &b, double alpha=1.);

/// Dot product between two vectors
double dot_product(d_vector const &a, d_vector const &b);

/// Conjugate gradient algorithm precnditioned with SSOR
void cg(double* val, int n_val, int* lines, int n_lines, int* cols, 
    int n_cols, double* rhs, int n_rhs, double* solution, int n_solution, 
    double tol);
