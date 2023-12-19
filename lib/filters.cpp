#include <Python.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <string>
#include <algorithm>

#include "numpy/arrayobject.h"

#ifdef _OPENMP
#include <omp.h>
#endif


inline int clip(int n, int lower, int upper)
{
    return std::max(lower, std::min(n, upper));
}


void _max_filter_horiz(float *src, float * dst, const int kernel_size, const int Nx, const int Ny){
#pragma omp parallel for
    for (int i = 0; i < Ny; i++){
        for (int j = 0; j < Nx; j++){
            float max = -1e10;
            for (int k = -kernel_size; k <= kernel_size; k++){
                const int j2 = clip(j + k, 0, Nx - 1);
                const float val = src[i * Nx + j2];
                if (val > max)
                    max = val;
            }
            dst[i * Nx + j] = max;
        }
    }
}

void _max_filter_vert(float *src, float * dst, const int kernel_size, const int Nx, const int Ny){
#pragma omp parallel for
    for (int j = 0; j < Nx; j++){
        for (int i = 0; i < Ny; i++){
            float max = -1e10;
            for (int k = -kernel_size; k <= kernel_size; k++){
                const int i2 = clip(i + k, 0, Ny - 1);
                const float val = src[i2 * Nx + j];
                if (val > max)
                    max = val;
            }
            dst[i * Nx + j] = max;
        }
    }
}


void _transpose(float *src, float *dst, const int Nx, const int Ny){
#pragma omp parallel for
    for (int j = 0; j < Nx; j++){
        for (int i = 0; i < Ny; i++){
            dst[j * Ny + i] = src[i * Nx + j];
        }
    }
}
static PyObject *c_maximum_filter_2d_float(PyObject *self, PyObject *args)
{

// #ifdef _OPENMP
//     const int nthreads = omp_get_max_threads();
//     std::cout << "Using " << nthreads << " thread(s)" << std::endl;
// #endif

// #ifdef __APPLE__
// #pragma omp parallel for
// #else
// #pragma omp parallel for schedule(dynamic)
// #endif
//     for (int i = 0; i < 32; i++){
//         std::cout << "Hello from thread " << omp_get_thread_num() << std::endl;
//     };

    PyArrayObject *src = NULL;
    int kernel_size;
    int max_threads;

    if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &src, &kernel_size, &max_threads))
        return NULL;

#ifdef _OPENMP
    omp_set_num_threads(max_threads);
#endif

    npy_intp *dims = PyArray_DIMS(src);
    const long Ny = dims[0];
    const long Nx = dims[1];

    PyArrayObject *dst = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT32);

    float *src_data = (float *)PyArray_DATA(src);
    float *dst_data = (float *)PyArray_DATA(dst);
    float *tmp = new float[Nx * Ny];

    _max_filter_horiz(src_data, tmp, kernel_size, Nx, Ny);
    _max_filter_vert(tmp, dst_data, kernel_size, Nx, Ny);

    // _max_filter_horiz(src_data, tmp, kernel_size, Nx, Ny);
    // _transpose(tmp, tmp2, Nx, Ny);
    // _max_filter_horiz(tmp2, tmp, kernel_size, Nx, Ny);
    // _transpose(tmp, dst_data, Nx, Ny);

    delete[] tmp;


    return PyArray_Return(dst);
}

//------------------------------------------------------------------------

static struct PyMethodDef methods[] = {
    {"c_maximum_filter_2d_float", c_maximum_filter_2d_float, METH_VARARGS, "point max filter"},
    {NULL, NULL, 0, NULL}

};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "filters",
    NULL,
    -1,
    methods,
    NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_filters(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
