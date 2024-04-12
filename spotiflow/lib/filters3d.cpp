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


void _max_filter_horiz(float *src, float * dst, const int kernel_size, const int Nx, const int Ny, const int Nz){
#pragma omp parallel for
    for (int i = 0; i < Ny; i++){
        for (int j = 0; j < Nx; j++){
            for (int k = 0; k < Nz; k++) {
                float max = -1e10;
                for (int l = -kernel_size; l <= kernel_size; l++){
                    const int j2 = clip(j + l, 0, Nx - 1);
                    const float val = src[k * Ny * Nx + i * Nx + j2];
                    if (val > max)
                        max = val;
                }
                dst[k * Ny * Nx + i * Nx + j] = max;
            }
        }
    }
}

void _max_filter_vert(float *src, float * dst, const int kernel_size, const int Nx, const int Ny, const int Nz) {
    #pragma omp parallel for
    for (int j = 0; j < Nx; j++){
        for (int i = 0; i < Ny; i++){
            for (int k = 0; k < Nz; k++) {
                float max = -1e10;
                for (int l = -kernel_size; l <= kernel_size; l++){
                    const int i2 = clip(i + l, 0, Ny - 1);
                    const float val = src[k * Ny * Nx + i2 * Nx + j];
                    if (val > max)
                        max = val;
                }
                dst[k * Ny * Nx + i * Nx + j] = max;
            }
        }
    }
}

void _max_filter_depth(float *src, float * dst, const int kernel_size, const int Nx, const int Ny, const int Nz) {
    #pragma omp parallel for
    for (int k = 0; k < Nz; k++){
        for (int i = 0; i < Ny; i++){
            for (int j = 0; j < Nx; j++){
                float max = -1e10;
                for (int l = -kernel_size; l <= kernel_size; l++){
                    const int k2 = clip(k + l, 0, Nz - 1);
                    const float val = src[k2 * Ny * Nx + i * Nx + j];
                    if (val > max)
                        max = val;
                }
                dst[k * Ny * Nx + i * Nx + j] = max;
            }
        }
    }
}



static PyObject *c_maximum_filter_3d_float(PyObject *self, PyObject *args)
{
    PyArrayObject *src = NULL;
    int kernel_size;
    int max_threads;

    if (!PyArg_ParseTuple(args, "O!ii", &PyArray_Type, &src, &kernel_size, &max_threads))
        return NULL;

#ifdef _OPENMP
    omp_set_num_threads(max_threads);
#endif

    npy_intp *dims = PyArray_DIMS(src);
    const long Nz = dims[0];
    const long Ny = dims[1];
    const long Nx = dims[2];

    PyArrayObject *dst = (PyArrayObject *)PyArray_SimpleNew(3, dims, NPY_FLOAT32);

    float *src_data = (float *)PyArray_DATA(src);
    float *dst_data = (float *)PyArray_DATA(dst);

    float *tmp1 = new float[Nx * Ny * Nz];
    float *tmp2 = new float[Nx * Ny * Nz];

    _max_filter_horiz(src_data, tmp1, kernel_size, Nx, Ny, Nz);
    _max_filter_vert(tmp1, tmp2, kernel_size, Nx, Ny, Nz);
    _max_filter_depth(tmp2, dst_data, kernel_size, Nx, Ny, Nz);


    delete[] tmp1;
    delete[] tmp2;


    return PyArray_Return(dst);
}

//------------------------------------------------------------------------

static struct PyMethodDef methods[] = {
    {"c_maximum_filter_3d_float", c_maximum_filter_3d_float, METH_VARARGS, "point max filter 3d"},
    {NULL, NULL, 0, NULL}

};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "filters3d",
    NULL,
    -1,
    methods,
    NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_filters3d(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
