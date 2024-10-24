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

#include <nanoflann.hpp>

inline int clip(int n, int lower, int upper)
{
    return std::max(lower, std::min(n, upper));
}

template <typename T> struct Point3D
{
        T x, y, z;
};

template <typename T>
struct PointCloud3D
{

    std::vector<Point3D<T>> pts;
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }
    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
};

inline int round_to_int(float r)
{
    return (int)lrint(r);
}


static PyObject *c_spotflow3d(PyObject *self, PyObject *args)
{

    PyArrayObject *points = NULL;
    PyArrayObject *dst = NULL;
    int shape_z, shape_y, shape_x;
    int grid_z, grid_y, grid_x;
    float scale;

    if (!PyArg_ParseTuple(args, "O!iiiiiif", &PyArray_Type, &points, &shape_z, &shape_y, &shape_x, &grid_z, &grid_y, &grid_x, &scale))
        return NULL;

    npy_intp *dims = PyArray_DIMS(points);

    npy_intp dims_dst[4];
    dims_dst[0] = shape_z / grid_z; // TODO: what if shape_z % grid_z != 0?
    dims_dst[1] = shape_y / grid_y; // TODO: what if shape_y % grid_y != 0?
    dims_dst[2] = shape_x / grid_x; // TODO: what if shape_x % grid_x != 0?
    dims_dst[3] = 4;

    dst = (PyArrayObject *)PyArray_SimpleNew(4, dims_dst, NPY_FLOAT32);

    // build kdtree

    PointCloud3D<float> cloud;
    float query_point[3];
    nanoflann::SearchParams params;
    std::vector<std::pair<size_t, float>> results;

    cloud.pts.resize(dims[0]);
    for (long i = 0; i < dims[0]; i++)
    {
        cloud.pts[i].z = *(float *)PyArray_GETPTR2(points, i, 0);
        cloud.pts[i].y = *(float *)PyArray_GETPTR2(points, i, 1);
        cloud.pts[i].x = *(float *)PyArray_GETPTR2(points, i, 2);
    }

    // construct a kd-tree:
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud3D<float>>,
        PointCloud3D<float>, 3 /* dim */>
        my_kd_tree_t;

    // build the index from points
    my_kd_tree_t index(3 /* dim */, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

    index.buildIndex();

    const float scale2 = scale * scale; // TODO: rescale?

#ifdef __APPLE__
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < dims_dst[0]; i++)
    {
        for (int j = 0; j < dims_dst[1]; j++)
        {
            for (int k = 0; k < dims_dst[2]; k++) 
            {

                // get the closest point
                const float query_pt[3] = {(float) grid_x*k, (float) grid_y*j, (float)grid_z*i};
                size_t ret_index;
                float out_dist_sqr;

                index.knnSearch(
                    &query_pt[0], 1, &ret_index, &out_dist_sqr);

                // the coords of the closest point
                const float px = cloud.pts[ret_index].x;
                const float py = cloud.pts[ret_index].y;
                const float pz = cloud.pts[ret_index].z;

                const float z = pz/grid_z - i;
                const float y = py/grid_y - j;
                const float x = px/grid_z - k;

                const float r2 = x * x + y * y + z * z;

                // the stereographic embedding
                const float x_prime = 2 * scale * x / (r2 + scale2);
                const float y_prime = 2 * scale * y / (r2 + scale2);
                const float z_prime = 2 * scale * z / (r2 + scale2);
                const float w_prime = -(r2 - scale2) / (r2 + scale2);

                *(float *)PyArray_GETPTR4(dst, i, j, k, 0) = w_prime;
                *(float *)PyArray_GETPTR4(dst, i, j, k, 1) = z_prime;
                *(float *)PyArray_GETPTR4(dst, i, j, k, 2) = y_prime;
                *(float *)PyArray_GETPTR4(dst, i, j, k, 3) = x_prime;
            }
        }
    }

    return PyArray_Return(dst);
}


static PyObject *c_gaussian3d(PyObject *self, PyObject *args)
{

    PyArrayObject *points = NULL;
    PyArrayObject *dst = NULL;
    PyArrayObject *sigmas = NULL;
    PyArrayObject *probs = NULL;
    int shape_z, shape_y, shape_x;
    int grid_z, grid_y, grid_x;

    if (!PyArg_ParseTuple(args, "O!O!O!iiiiii", &PyArray_Type, &points, &PyArray_Type, &probs, &PyArray_Type, &sigmas, &shape_z, &shape_y, &shape_x, &grid_z, &grid_y, &grid_x))
        return NULL;

    npy_intp *dims = PyArray_DIMS(points);

    npy_intp dims_dst[3];
    dims_dst[0] = shape_z / grid_z; // TODO: what if shape_z % grid_z != 0?
    dims_dst[1] = shape_y / grid_y; // TODO: what if shape_y % grid_y != 0?
    dims_dst[2] = shape_x / grid_x; // TODO: what if shape_x % grid_x != 0?

    dst = (PyArrayObject *)PyArray_SimpleNew(3/* dim */, dims_dst, NPY_FLOAT32);


    // build kdtree

    PointCloud3D<float> cloud;
    float query_point[3];
    nanoflann::SearchParams params;
    std::vector<std::pair<size_t, float>> results;

    cloud.pts.resize(dims[0]);
    for (long i = 0; i < dims[0]; i++)
    {
        cloud.pts[i].z = *(float *)PyArray_GETPTR2(points, i, 0);
        cloud.pts[i].y = *(float *)PyArray_GETPTR2(points, i, 1);
        cloud.pts[i].x = *(float *)PyArray_GETPTR2(points, i, 2);
    }

    // construct a kd-tree:
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud3D<float>>,
        PointCloud3D<float>, 3>
        my_kd_tree_t;

    // build the index from points
    my_kd_tree_t index(3/* dim */, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

    index.buildIndex();


    #ifdef __APPLE__
    #pragma omp parallel for
    #else
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i = 0; i < dims_dst[0]; i++)
    {
        for (int j = 0; j < dims_dst[1]; j++)
        {
            for (int k = 0; k < dims_dst[2]; k++)
            {

                // get the closest point
                const float query_pt[3] = {(float) grid_x*k, (float) grid_y*j, (float) grid_z*i};
                size_t ret_index;
                float out_dist_sqr;

                index.knnSearch(
                    &query_pt[0], 1, &ret_index, &out_dist_sqr);

                // the coords of the closest point
                const float px = cloud.pts[ret_index].x;
                const float py = cloud.pts[ret_index].y;
                const float pz = cloud.pts[ret_index].z;

                const float z = floor(pz/grid_z) - i;
                const float y = floor(py/grid_y) - j;
                const float x = floor(px/grid_x) - k;

                const float r2 = x * x + y * y + z * z;

                const float prob = *(float *)PyArray_GETPTR1(probs, ret_index);
                const float sigma = *(float *)PyArray_GETPTR1(sigmas, ret_index);
                const float sigma_denom = 2 * sigma * sigma / cbrt(grid_z * grid_y * grid_x);

                // the gaussian value
                const float val = prob * exp(-r2 / sigma_denom);

                *(float *)PyArray_GETPTR3(dst, i, j, k) = val;
            }
        }
    }

    return PyArray_Return(dst);
}

//------------------------------------------------------------------------

static struct PyMethodDef methods[] = {
    {"c_spotflow3d", c_spotflow3d, METH_VARARGS, "spot flow 3d"},
    {"c_gaussian3d", c_gaussian3d, METH_VARARGS, "gaussian 3d"},
    {NULL, NULL, NULL, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spotflow3d",
    NULL,
    -1,
    methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_spotflow3d(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
