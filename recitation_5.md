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

`PyCUDA` provides a handful of high level API's for _gpu resource allocation_,
_data transfer_, and _cuda code compilation_, etc. In the following, we will
go through a toy program [pycuda_demo.py](./src/pycuda_demo.py) to demostrate
some `PyCUDA` API's. We will see that `PyCUDA` does a lot for us, and some of
the parts are implicitly done by `PyCUDA`. 

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

To start a GPU program, we usually need to go through 3 steps: i) initialize a cuda driver;
ii) specify a GPU device to the driver; iii) create a context on the device. However, we
are lazy, and `PyCUDA` knows _that_. `PyCUDA` allows us to achieve the three steps in one
line:

    import pycuda.autoinit
    import pycuda.driver as drv

### Source Code and Compilation ###

`SourceModule` is a python wrapper of the cuda compiler. It takes the cuda source
code stored in a python string together with some optinal arguments as input, and
compiles the source code, and caches the resultant program, so later on we run the
program, the `PyCUDA` compiler will not re-compile the source code, unless it is
updated.

    from pycuda.compiler import SourceModule
    mod = SourceModule("""
    __global__ void vec_elm_mul(float *dest, float *a, float *b)
    {
        const int i = threadIdx.x;
        dest[i] = a[i] * b[i];
    }
    """, options = ["--ptxas-options=-v"])
    vec_elm_mul = mod.get_function("vec_elm_mul")

With the optianl argument `options = ["--ptxas-options=-v"]`, the compiler prints
out compilation message at the ternimal output. For example, after running
`pycuda_demo.py`, we will see,

    pycuda_demo.py:13: UserWarning: The CUDA compiler succeeded, but said the following:
    ptxas info    : 0 bytes gmem
    ptxas info    : Compiling entry function 'multiply_them' for 'sm_20'
    ptxas info    : Function properties for multiply_them
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    ptxas info    : Used 12 registers, 56 bytes cmem[0]
    
### Initialization of data on CPU ###

Here is quite standard. We use `numpy` to allocate memory on CPU.

    a = numpy.random.randn(400).astype(numpy.float32)
    b = numpy.random.randn(400).astype(numpy.float32)
    dest = numpy.zeros_like(a)

### Executation and More ###

At first glance, it seems that we ommit the memory transfer from CPU to GPU and
the other direction. In fact, we call two `PyCUDA` magical functions
`pycuda.driver.In` and `pycuda.driver.Out` to perform data copying. When an CPU
array is passed into `pycuda.driver.In` and the output of `pycuda.driver.In` is
then passed into a cuda function, `PyCUDA` will automatically allocate a chunk
of memory on GPU, copy the data from CPU to GPU, and finally pass the address
of the GPU memroy to the cuda function.

    vec_elm_mul( drv.Out(dest), drv.In(a), drv.In(b),
                 block=(400,1,1), grid=(1,1))
                 
Likewise, `pycuda.driver.Out` will allocate memory on GPU first, pass the address
of GPU memory to the cuda function, and copy the data from GPU to CPU after the
execution of the cuda function.
