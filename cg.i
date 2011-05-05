%module cg

%{
    #define SWIG_FILE_WITH_INIT
    #include "cg.hh"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (int* INPLACE_ARRAY1, int DIM1) {(int* lines, int n_lines),(int* cols,int n_cols)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* val, int n_val),(double* rhs, int n_rhs),(double* solution, int n_solution)}

%include "cg.hh"
