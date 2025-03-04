"""Adapted from https://github.com/stardist/stardist/blob/master/setup.py"""
from __future__ import absolute_import, print_function
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from numpy import get_include
from os import path

# ------------------------------------------------------------------------------------
_dir = path.dirname(__file__)


class build_ext_openmp(build_ext):
    # https://www.openmp.org/resources/openmp-compilers-tools/
    # python setup.py build_ext --help-compiler
    openmp_compile_args = {
        "msvc": [["/openmp"]],
        "intel": [["-qopenmp"]],
        "*": [["-fopenmp"], ["-Xpreprocessor", "-fopenmp"]],
    }
    openmp_link_args = openmp_compile_args  # ?

    def build_extension(self, ext):
        compiler = self.compiler.compiler_type.lower()
        if compiler.startswith("intel"):
            compiler = "intel"
        if compiler not in self.openmp_compile_args:
            compiler = "*"

        # thanks to @jaimergp (https://github.com/conda-forge/staged-recipes/pull/17766)
        # issue: qhull has a mix of c and c++ source files
        #        gcc warns about passing -std=c++11 for c files, but clang errors out
        compile_original = self.compiler._compile

        def compile_patched(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # remove c++ specific (extra) options for c files
            if src.lower().endswith(".c"):
                extra_postargs = [
                    arg for arg in extra_postargs if not arg.lower().startswith("-std")
                ]
            return compile_original(obj, src, ext, cc_args, extra_postargs, pp_opts)

        # monkey patch the _compile method
        self.compiler._compile = compile_patched

        # store original args
        _extra_compile_args = list(ext.extra_compile_args)
        _extra_link_args = list(ext.extra_link_args)

        # try compiler-specific flag(s) to enable openmp
        for compile_args, link_args in zip(
            self.openmp_compile_args[compiler], self.openmp_link_args[compiler]
        ):

            try:
                ext.extra_compile_args = _extra_compile_args + compile_args
                ext.extra_link_args = _extra_link_args + link_args
                print(">>> try building with OpenMP support: ", compile_args, link_args)
                return super(build_ext_openmp, self).build_extension(ext)
            except Exception as _:
                print(f">>> compiling with '{' '.join(compile_args)}' failed")

        print(">>> compiling with OpenMP support failed, re-trying without")

        ext.extra_compile_args = _extra_compile_args
        ext.extra_link_args = _extra_link_args
        return super(build_ext_openmp, self).build_extension(ext)


external_root = path.join(_dir, "spotiflow", "lib", "external")
nanoflann_root = path.join(external_root, "nanoflann")

setup(
    cmdclass={"build_ext": build_ext_openmp},
    ext_modules=[
        Extension(
            "spotiflow.lib.spotflow2d",
            sources=["spotiflow/lib/spotflow2d.cpp"],
            extra_compile_args=["-std=c++11"],
            include_dirs=[get_include()] + [nanoflann_root],
        ),
        Extension(
            "spotiflow.lib.spotflow3d",
            sources=["spotiflow/lib/spotflow3d.cpp"],
            extra_compile_args=["-std=c++11"],
            include_dirs=[get_include()] + [nanoflann_root],
        ),
        Extension(
            "spotiflow.lib.point_nms",
            sources=["spotiflow/lib/point_nms.cpp"],
            extra_compile_args=["-std=c++11"],
            include_dirs=[get_include()] + [nanoflann_root],
        ),
        Extension(
            "spotiflow.lib.point_nms3d",
            sources=["spotiflow/lib/point_nms3d.cpp"],
            extra_compile_args=["-std=c++11"],
            include_dirs=[get_include()] + [nanoflann_root],
        ),
        Extension(
            "spotiflow.lib.filters",
            sources=["spotiflow/lib/filters.cpp"],
            extra_compile_args=["-std=c++11"],
            include_dirs=[get_include()] + [nanoflann_root],
        ),
        Extension(
            "spotiflow.lib.filters3d",
            sources=["spotiflow/lib/filters3d.cpp"],
            extra_compile_args=["-std=c++11"],
            include_dirs=[get_include()] + [nanoflann_root],
        ),
    ],
    include_package_data=True,
)
