# Recitation 5 #

## PyCUDA: First Look ##

`PyCUDA` is a wrapper for `CUDA` which hides detail operations needed to compile
and run a cuda program. In general, a PyCUDA program contain can be decomposed
into 6 parts as follows,

  * GPU resource allocation
  * Specifying cuda source code, and compilation.
  * Initialization of data on CPU
  * Copying data from CPU to GPU
  * Executation of CUDA program
  * Copying data back from GPU to CPU


    import pycuda.autoinit
    import pycuda.driver as drv
    import numpy

    from pycuda.compiler import SourceModule

    mod = SourceModule("""
    __global__ void vec_elm_mul(float *dest, float *a, float *b)
    {
        const int i = threadIdx.x;
        dest[i] = a[i] * b[i];
    }
    """, options = ["--ptxas-options=-v"])
    vec_elm_mul = mod.get_function("vec_elm_mul")

    a = numpy.random.randn(400).astype(numpy.float32)
    b = numpy.random.randn(400).astype(numpy.float32)

    dest = numpy.zeros_like(a)

### GPU resource allocation ###

    import pycuda.autoinit
    import pycuda.driver as drv

### Source Code and Compilation ###

    from pycuda.compiler import SourceModule
    mod = SourceModule("""
    mod = SourceModule("""
    __global__ void vec_elm_mul(float *dest, float *a, float *b)
    {
        const int i = threadIdx.x;
        dest[i] = a[i] * b[i];
    }
    """, options = ["--ptxas-options=-v"])
    vec_elm_mul = mod.get_function("vec_elm_mul")

### Initialization of data on CPU ###

    a = numpy.random.randn(400).astype(numpy.float32)
    b = numpy.random.randn(400).astype(numpy.float32)
    dest = numpy.zeros_like(a)

### Executation ###

    vec_elm_mul( drv.Out(dest), drv.In(a), drv.In(b),
                 block=(400,1,1), grid=(1,1))
