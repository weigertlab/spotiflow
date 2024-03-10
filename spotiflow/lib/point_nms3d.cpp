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


static PyObject *c_point_nms_3d(PyObject *self, PyObject *args)
{

    PyArrayObject *points = NULL;
    float min_distance;

    if (!PyArg_ParseTuple(args, "O!f", &PyArray_Type, &points, &min_distance))
        return NULL;

    npy_intp *dims = PyArray_DIMS(points);
    const float min_distance_squared = min_distance * min_distance;
    const long n_points = dims[0];

    npy_intp dims_dst[1];
    dims_dst[0] = n_points;
    PyArrayObject *dst = (PyArrayObject *)PyArray_SimpleNew(1, dims_dst, NPY_BOOL);


    // std::cout << "dims[0]: " << dims[0] << std::endl;
    // std::cout << "dims[1]: " << dims[1] << std::endl;
    // std::cout << "min_distance: " << min_distance << std::endl;


    // build kdtree

    PointCloud3D<float> cloud;

    cloud.pts.resize(dims[0]);
    for (long i = 0; i < n_points; i++)
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
    my_kd_tree_t index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

    index.buildIndex();


// #ifdef __APPLE__
// #pragma omp parallel for
// #else
// #pragma omp parallel for schedule(dynamic)
// #endif

    // for (long k = 0; k < n_points; k++)
    // {
    //     const float x = index.dataset.kdtree_get_pt(k,0);
    //     const float y = index.dataset.kdtree_get_pt(k,1);

    //     // if (k != i){
    //     std::cout << "Index: " << k << " (y,x) = "<< y << " " << x << std::endl;
    // }


    bool * suppressed = new bool[n_points];
    for (long i = 0; i < n_points; i++)
    {
        suppressed[i] = false;
    }

    std::vector<std::pair<size_t, float>> results;
    float query_point[3];
    nanoflann::SearchParams params;

    for (long i = 0; i < n_points; i++)
    {
        if (suppressed[i]){
            continue;
        }
        query_point[0] = *(float *)PyArray_GETPTR2(points, i, 2);
        query_point[1] = *(float *)PyArray_GETPTR2(points, i, 1);
        query_point[2] = *(float *)PyArray_GETPTR2(points, i, 0);
        std::vector<std::pair<size_t, float>> ret_matches;
        const size_t n_matches = index.radiusSearch(&query_point[0], min_distance_squared, ret_matches, params);

        // std::cout << "----- " << i << "  (y,x) = " << query_point[0] << ", " << query_point[1] << " n_matches: " << n_matches << std::endl;

        for (long j = 0; j < n_matches; j++)
        {
            const long k = ret_matches[j].first;
            const float dist = ret_matches[j].second;
            if ((k != i)  && (dist < min_distance_squared)) {
                // std::cout << "suppressed: " << k << " "<< *(float *)PyArray_GETPTR2(points, k, 0) << " (y,x) = " << *(float *)PyArray_GETPTR2(points, k, 0) << " distance " << dist << std::endl;
                suppressed[k] = true;
            }
        }


    }

    for (long i = 0; i < n_points; i++)
    {
        *(bool *)PyArray_GETPTR1(dst, i) = !suppressed[i];
    }

    delete [] suppressed;
    return PyArray_Return(dst);
}

//------------------------------------------------------------------------

static struct PyMethodDef methods[] = {
    {"c_point_nms_3d", c_point_nms_3d, METH_VARARGS, "point nms 3D"},
    {NULL, NULL, 0, NULL}

};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "point_nms3d",
    NULL,
    -1,
    methods,
    NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_point_nms3d(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
