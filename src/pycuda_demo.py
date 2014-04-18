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

d = numpy.ones(2,numpy.int32)

grid = (int(d[0]),int(d[1]))
print grid

vec_elm_mul( drv.Out(dest), drv.In(a), drv.In(b),
                    block=(400,1,1), grid=grid)

print dest-a*b
