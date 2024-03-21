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

template <typename T> struct Point2D
{
        T x, y;
};

template <typename T> struct Point3D
{
        T x, y, z;
};

template <typename T> bool points_greater(const Point3D<T> &a, const Point3D<T> &b)
{
    return a.z>b.z;
}

template <typename T>
struct PointCloud2D
{

    std::vector<Point2D<T>> pts;
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else
            return pts[idx].y;
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


static PyObject *c_gaussian2d(PyObject *self, PyObject *args)
{

    PyArrayObject *points = NULL;
    PyArrayObject *probs = NULL;
    PyArrayObject *sigmas = NULL;
    PyArrayObject *dst = NULL;
    int shape_y, shape_x;
    float sigma;

    if (!PyArg_ParseTuple(args, "O!O!O!ii", &PyArray_Type, &points, &PyArray_Type, &probs, &PyArray_Type, &sigmas, &shape_y, &shape_x))
        return NULL;

    npy_intp *dims = PyArray_DIMS(points);

    npy_intp dims_dst[2];
    dims_dst[0] = shape_y;
    dims_dst[1] = shape_x;

    dst = (PyArrayObject *)PyArray_SimpleNew(2, dims_dst, NPY_FLOAT32);


    // build kdtree

    PointCloud2D<float> cloud;
    float query_point[2];
    nanoflann::SearchParams params;
    std::vector<std::pair<size_t, float>> results;

    cloud.pts.resize(dims[0]);
    for (long i = 0; i < dims[0]; i++)
    {
        cloud.pts[i].y = *(float *)PyArray_GETPTR2(points, i, 0);
        cloud.pts[i].x = *(float *)PyArray_GETPTR2(points, i, 1);
    }

    // construct a kd-tree:
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud2D<float>>,
        PointCloud2D<float>, 2>
        my_kd_tree_t;

    // build the index from points
    my_kd_tree_t index(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

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

            // get the closest point
            const float query_pt[2] = {(float)j, (float)i};
            size_t ret_index;
            float out_dist_sqr;

            index.knnSearch(
                &query_pt[0], 1, &ret_index, &out_dist_sqr);

            // the coords of the closest point
            const float px = cloud.pts[ret_index].x;
            const float py = cloud.pts[ret_index].y;

            const float prob = *(float *)PyArray_GETPTR1(probs, ret_index);
            const float sigma = *(float *)PyArray_GETPTR1(sigmas, ret_index);
            const float sigma_denom = 2 * sigma * sigma;

            const float y = py - i;
            const float x = px - j;

            const float r2 = x * x + y * y;

            // the gaussian value
            const float val = prob*exp(-r2 / sigma_denom);

            // const float val = 0;

            *(float *)PyArray_GETPTR2(dst, i, j) = val;
        }
    }

    return PyArray_Return(dst);
}


static PyObject *c_spotflow2d(PyObject *self, PyObject *args)
{

    PyArrayObject *points = NULL;
    PyArrayObject *dst = NULL;
    int shape_y, shape_x;
    float scale;

    if (!PyArg_ParseTuple(args, "O!iif", &PyArray_Type, &points, &shape_y, &shape_x, &scale))
        return NULL;

    // #ifdef _OPENMP
    // std::cout << "OpenMP is enabled" << std::endl;
    // #else
    // std::cout << "OpenMP is disabled" << std::endl;
    // #endif

    npy_intp *dims = PyArray_DIMS(points);

    npy_intp dims_dst[3];
    dims_dst[0] = shape_y;
    dims_dst[1] = shape_x;
    dims_dst[2] = 3;

    dst = (PyArrayObject *)PyArray_SimpleNew(3, dims_dst, NPY_FLOAT32);

    // build kdtree

    PointCloud2D<float> cloud;
    float query_point[2];
    nanoflann::SearchParams params;
    std::vector<std::pair<size_t, float>> results;

    cloud.pts.resize(dims[0]);
    for (long i = 0; i < dims[0]; i++)
    {
        cloud.pts[i].y = *(float *)PyArray_GETPTR2(points, i, 0);
        cloud.pts[i].x = *(float *)PyArray_GETPTR2(points, i, 1);
    }

    // construct a kd-tree:
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud2D<float>>,
        PointCloud2D<float>, 2>
        my_kd_tree_t;

    // build the index from points
    my_kd_tree_t index(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

    index.buildIndex();

    const float scale2 = scale * scale;

#ifdef __APPLE__
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < dims_dst[0]; i++)
    {
        for (int j = 0; j < dims_dst[1]; j++)
        {

            // get the closest point
            const float query_pt[2] = {(float)j, (float)i};
            size_t ret_index;
            float out_dist_sqr;

            index.knnSearch(
                &query_pt[0], 1, &ret_index, &out_dist_sqr);

            // the coords of the closest point
            const float px = cloud.pts[ret_index].x;
            const float py = cloud.pts[ret_index].y;

            const float y = py - i;
            const float x = px - j;

            const float r2 = x * x + y * y;

            // the stereographic embedding
            const float x_prime = 2 * scale * x / (r2 + scale2);
            const float y_prime = 2 * scale * y / (r2 + scale2);
            const float z_prime = -(r2 - scale2) / (r2 + scale2);

            *(float *)PyArray_GETPTR3(dst, i, j, 0) = z_prime;
            *(float *)PyArray_GETPTR3(dst, i, j, 1) = y_prime;
            *(float *)PyArray_GETPTR3(dst, i, j, 2) = x_prime;
        }
    }

    return PyArray_Return(dst);
}


float interp_flow(PyArrayObject *data, const int dim, float y, float x, int Ny, int Nx)
{

    if (x < 0 || x >= Nx || y < 0 || y >= Ny)
        return 0;

    int x0 = (int)floor(x);
    int x1 = x0+1;
    int y0 = (int)floor(y);
    int y1 = y0+1;

    float dx = x - x0;
    float dy = y - y0;

    float v00 = *(float *)PyArray_GETPTR3(data, y0, x0, dim);
    float v01 = *(float *)PyArray_GETPTR3(data, y0, x1, dim);
    float v10 = *(float *)PyArray_GETPTR3(data, y1, x0, dim);
    float v11 = *(float *)PyArray_GETPTR3(data, y1, x1, dim);

    float v0 = v00 * (1 - dx) + v01 * dx;
    float v1 = v10 * (1 - dx) + v11 * dx;

    return v0 * (1 - dy) + v1 * dy;
}

static PyObject *c_cluster_flow2d(PyObject *self, PyObject *args)
{

    PyArrayObject *points = NULL;
    PyArrayObject *flow = NULL;
    PyArrayObject *dst_mapped = NULL;
    PyArrayObject *dst = NULL;
    float dt;
    int steps;
    float atol;
    float min_distance;

    if (!PyArg_ParseTuple(args, "O!O!fffi", &PyArray_Type, &points, &PyArray_Type, &flow, &dt, &atol, &min_distance, &steps))
        return NULL;

    npy_intp *dims_points = PyArray_DIMS(points);
    npy_intp *dims_flow = PyArray_DIMS(flow);

    std::vector<Point3D<float>> coords;

    for (long i = 0; i < dims_points[0]; i++)
    {
        Point3D<float> pp;
        pp.y = *(float *)PyArray_GETPTR2(points, i, 0);
        pp.x = *(float *)PyArray_GETPTR2(points, i, 1);
        pp.z = interp_flow(flow, 0, pp.y, pp.x, dims_flow[0], dims_flow[1]);
        coords.push_back(pp);
    }

    std::vector<bool> suppressed(coords.size(), false);


#ifdef __APPLE__
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic)
#endif
    for (long i = 0; i < coords.size(); i++){
        float py = coords[i].y;
        float px = coords[i].x;

        float dx, dy;

        for (long n = 0; n < steps; n++)
        {

            float vz = interp_flow(flow, 0, py, px, dims_flow[0], dims_flow[1]);
            float vy = interp_flow(flow, 1, py, px, dims_flow[0], dims_flow[1]);
            float vx = interp_flow(flow, 2, py, px, dims_flow[0], dims_flow[1]);

            vx = vx / (1 + vz);
            vy = vy / (1 + vz);

            // if (fabs(coords[i].x-16)<1 && fabs(coords[i].y-13)<1)
            //     std::cout << "i: " << i <<   " vx: " << vx << ", vy: " << vy << std::endl;

            dy = vy * dt;
            dx = vx * dt;

            py += dy;
            px += dx;

            if (dx * dx + dy * dy <= atol * atol)
                break;
        }

        if (dx * dx + dy * dy > atol * atol){
            suppressed[i] = true;
        }

        coords[i].y = py;
        coords[i].x = px;

    }

    npy_intp dims_dst_mapped[2];
    dims_dst_mapped[0] = coords.size();
    dims_dst_mapped[1] = 2;

    dst_mapped = (PyArrayObject *)PyArray_SimpleNew(2, dims_dst_mapped, NPY_FLOAT32);

    for (long i = 0; i < coords.size(); i++)
    {
        *(float *)PyArray_GETPTR2(dst_mapped, i, 0) = coords[i].y;
        *(float *)PyArray_GETPTR2(dst_mapped, i, 1) = coords[i].x;
    }


    // average weighted aggregation
    // sort points by z

    std::sort(coords.begin(), coords.end(), points_greater<float>);

    // for (long i = 0; i < coords.size(); i++){
    //     std::cout << "z" << coords[i].z << " y: " << coords[i].y << " x: " << coords[i].x << std::endl;
    // }


    for (long i = 0; i < coords.size(); i++){
        if (suppressed[i])
            continue;


        // float new_weight = 1+coords[i].z;
        float new_weight = 1;
        Point2D<float> new_pos ;
        new_pos.x = new_weight * coords[i].x;
        new_pos.y = new_weight * coords[i].y;

        // std::cout << "i:   " << i << " coords[i].x: " << coords[i].x << " coords[i].y: " << coords[i].y << std::endl;

// #ifdef __APPLE__
// #pragma omp parallel for reduction(+:new_weight) reduction(+:new_pos)
// #else
// #pragma omp parallel for schedule(dynamic) reduction(+:new_weight) reduction(+:new_pos)
// #endif
        for (long j = i + 1; j < coords.size(); j++){
            float dy = coords[i].y - coords[j].y;
            float dx = coords[i].x - coords[j].x;

            if (dy * dy + dx * dx < min_distance * min_distance){
                suppressed[j] = true;

                new_pos.x += coords[j].x;
                new_pos.y += coords[j].y;
                // new_weight += 1+coords[j].z;
                new_weight += 1;

                // std::cout << new_weight << std::endl;
            }
        }



        // new point is weighted average of all points within min_distance
        coords[i].x = new_pos.x / new_weight;
        coords[i].y = new_pos.y / new_weight;

        // std::cout << "---> " << i << " coords[i].x: " << coords[i].x << " coords[i].y: " << coords[i].y << std::endl;

    }

    std::vector<Point3D<float>> coords_filtered;

    for (long i = 0; i < coords.size(); i++){
        if (!suppressed[i])
            coords_filtered.push_back(coords[i]);
    }

    npy_intp dims_dst[2];
    dims_dst[0] = coords_filtered.size();
    dims_dst[1] = 2;

    dst = (PyArrayObject *)PyArray_SimpleNew(2, dims_dst, NPY_FLOAT32);

    for (long i = 0; i < coords_filtered.size(); i++)
    {
        *(float *)PyArray_GETPTR2(dst, i, 0) = coords_filtered[i].y;
        *(float *)PyArray_GETPTR2(dst, i, 1) = coords_filtered[i].x;
    }



    PyObject *ret = PyTuple_New(2);
    PyTuple_SetItem(ret, 0, PyArray_Return(dst));
    PyTuple_SetItem(ret, 1, PyArray_Return(dst_mapped));

    return PyTuple_Pack(2, PyArray_Return(dst), PyArray_Return(dst_mapped));
}

//------------------------------------------------------------------------

static struct PyMethodDef methods[] = {
    {"c_spotflow2d", c_spotflow2d, METH_VARARGS, "spot flow"},
    {"c_gaussian2d", c_gaussian2d, METH_VARARGS, "gaussian"},
    {"c_cluster_flow2d", c_cluster_flow2d, METH_VARARGS, "cluster flow"},
    {NULL, NULL, 0, NULL}

};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spotflow2d",
    NULL,
    -1,
    methods,
    NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit_spotflow2d(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}