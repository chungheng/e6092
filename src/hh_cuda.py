"""
Suject   : Recitation 5
Detaul   : Use PyCUDA to simulate one Hodgkin-Huxley Neuron
Execution: type "python hh_cuda.py" in shell
Author   : Chung-Heng Yeh <chyeh@ee.columbia.edu>
"""
# Use device 4 on huxley.ee.columbia.edu
import pycuda.driver as driver
driver.init()
myDev = driver.Device(0)
myCtx = myDev.make_context()

import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import numpy as np
import progressbar as pb

# Use atexit to automatically pop the device context atd  the end of execution
import atexit
atexit.register(myCtx.pop)

# Set up matplotlib configuration
import matplotlib as mp
mp.rc('savefig', dpi=300)
mp.use('AGG')
import pylab as p


# GPU kernel function for simulation
mod = SourceModule("""
__device__ float alpha_nmh(int idx, float v)
{
    float a = 0;
    if( idx == 1 ) //n
        a =  (v+55.0)/(-100.0*(expf(-(v+55.0)/10.0)-1.0));
    else if( idx == 2 ) //m
        a = (v+40.0)/(-10.0*(expf(-(v+40.0)/10.0)-1.0));
    else if( idx == 3 ) //h
        a = 7.0*expf(-(v+65.0)/20.0)/100.0;
    return a;
}
__device__ float beta_nmh(int idx, float v)
{
    float b = 0;
    if( idx == 1 ) //n
        b =  (v+55.0)/(-100.0*(expf(-(v+55.0)/10.0)-1.0));
    else if( idx == 2 ) //m
        b = (v+40.0)/(-10.0*(expf(-(v+40.0)/10.0)-1.0));
    else if( idx == 3 ) //h
        b = 7.0*expf(-(v+65.0)/20.0)/100.0;
    return b;
}
__global__ void cuda_gpu_simulate(
     int num_iter, float dt,
     float *x0, float *x, float *I_ext )
{
    const int idx = threadIdx.x;
    int xidx = idx;
    float I;
    float E[3] = {-77.0,  50.0, -54.387};
    float g[3] = { 36.0, 120.0,     0.3};
    float a,b;
    float n,m,h,v;

    if(idx<4)
        x[idx] = x0[idx];

    for(int i=1; i<num_iter; ++i)
    {
        xidx = xidx+4;

        // Apply Runge-Kutta Method
        // 1st round
        v = x[4*i-4];
        I = I_ext[i-1];
        a = alpha_nmh(idx,v);
        b = beta_nmh(idx,v);
        // Branch: V and (n,m,h) are calculated seperately
        if( idx == 0 )
        {
            n = x[xidx-3];
            m = x[xidx-2];
            h = x[xidx-1];
            x[xidx] = -g[0]*n*n*n*n*(v-E[0])
                      -g[1]*m*m*m*h*(v-E[1])
                      -g[2]*(v-E[2])+I;
        }
        if( 0<idx )
        {
            float temp = x[xidx-4];
            x[xidx] = a*(1-temp) - b*temp;
        }
        x[xidx] = dt * 0.5 * x[xidx] + x[xidx-4];

        // 2nd round
        v = x[4*i];
        I = 0.5*(I+I_ext[i]);
        if( idx == 1 ) //n
        {
            a = (v+55.0)/(-100.0*(expf(-(v+55.0)/10.0)-1.0)); // m
            b = 1.0*expf(-(v+65.0)/80.0)/8.0;
        }
         else if( idx == 2 ) //m
        {
            a = (v+40.0)/(-10.0*(expf(-(v+40.0)/10.0)-1.0));
            b = 4.0*expf(-(v+65.0)/18.0);
        }
        else if( idx == 3 )
        {
            a = 7.0*expf(-(v+65.0)/20.0)/100.0;             // h
            b = 1.0/(expf(-(v+35.0)/10.0)+1.0);
        }
        // Branch: V and (n,m,h) are calculated seperately
        if( idx == 0 )
        {
            n = x[xidx+1];
            m = x[xidx+2];
            h = x[xidx+3];
            x[xidx] = -g[0]*n*n*n*n*(v-E[0])
                          -g[1]*m*m*m*h*(v-E[1])
                          -g[2]*(v-E[2])+I;
        }
        if( 0<idx )
        {
            float temp = x[xidx];
            x[xidx] = a*(1-temp) - b*temp;
        }
        x[xidx] = dt * x[xidx] + x[xidx-4];
    }
}
""", options = ["--ptxas-options=-v"])

class Hodgkin_Huxley_Neuron:
    """
    Hodgkin-Huxley Neuron

    Parameters:
    -----------
    x0 : Initial value of V, n, m, h; The default setting is
         (V,n,m,h) = (-65, 0.5, 0.2, 0.6);

    Functions:
    ----------
    cpu_simulate(t, I_ext):
        Simulate the neuron with injected current I_ext on the CPU during the
        time sequence t.

    gpu_simulate(t, I_ext):
        Similar to cpu_simulate(), however, the simulation is run on the GPU.

    After simulation, use Hodgkin_Huxley_Neuron.V(N/M/H) to access the
    simulation results.
    """
    E = np.array([-77.,  50., -54.387])
    g = np.array([ 36., 120.,     0.3])

    alpha_m = lambda self, V: (V+55.)/(-100.*np.exp(-(V+55.)/10.)-1.)
    alpha_n = lambda self, V: (V+40.)/(-10.*np.exp(-(V+40./10.))-1.)
    alpha_h = lambda self, V: 7.*np.exp(-(V+65.)/20.)/100.

    beta_m = lambda self, V: 1.*np.exp(-(V+65.)/80.0)/8.
    beta_n = lambda self, V: 4.*np.exp(-(V+65.)/18.)
    beta_h = lambda self, V: 1./(np.exp(-(V+35.)/10.)+1.)

    f = lambda self, x, I: np.array(
                  [-self.g[0]*x[1]**4*(x[0]-self.E[0])
                      -self.g[1]*x[2]**3*x[3]*(x[0]-self.E[1])
                      -self.g[2]*(x[0]-self.E[2])+I,
                   self.alpha_n(x[0])*(1-x[1])-self.beta_n(x[0])*x[1],
                   self.alpha_m(x[0])*(1-x[2])-self.beta_m(x[0])*x[2],
                   self.alpha_h(x[0])*(1-x[3])-self.beta_h(x[0])*x[3]])

    cuda_gpu_simulate = mod.get_function("cuda_gpu_simulate")

    def __init__(self, x0=[-65, 0.5, 0.2, 0.6]):

        # Initialize state matrix, x will be n by 4 matrix
        # x[:,i] for i = 0,1,2,3 stores the values of V, n. m. and h.
        self.x0 = np.array(x0).astype(np.float32)
        self.x  = []
        self.dt = []
        self.n  = []

    def cpu_interation(self, idx, I1, I2):
        I2 = 0.5 * (I1+I2)
        k1 = self.dt * self.f(self.x[:, idx-1], I1)
        k2 = self.dt * self.f(self.x[:, idx-1]+0.5*k1, I2)
        self.x[:,idx] = self.x[:,idx-1]+k2

    def cpu_simulate_init(self, t):
        self.n  = len(t)
        self.dt = 1000*(t[1]-t[0])
        self.x  = np.empty((4,self.n), np.float64)

    def cpu_simulate_finish(self):
        self.V = self.x[0]
        self.N = self.x[1]
        self.M = self.x[2]
        self.H = self.x[3]

    def cpu_simulate(self, t, I_ext):
        self.cpu_simulate_init(t)
        pbar = pb.ProgressBar(maxval=self.n).start()
        self.x[:,0] = self.x0
        for i in xrange(1,self.n):
            pbar.update(i)
            self.cpu_interation(i,I_ext[i-1],I_ext[i])
        pbar.finish()
        cpu_simulate_finish()

    def gpu_simulate_init(self, t, I_ext):
        self.n  = len(t)
        self.dt = 1000*(t[1]-t[0])
        self.x  = np.empty(4*self.n, np.float32)
        self.g_x = gpuarray.empty((4*self.n,),np.float64)
        self.g_I = gpuarray.to_gpu(I_ext)
        self.g_x0 = gpuarray.to_gpu(np.array(self.x0))

    def gpu_simulate_finish(self):
        self.x = self.g_x.get()
        self.I = self.g_I.get()
        self.V = self.x[0:4*self.n:4]
        self.N = self.x[1:4*self.n:4]
        self.M = self.x[2:4*self.n:4]
        self.H = self.x[3:4*self.n:4]
        #:pdb.set_trace()

    def gpu_simulate(self, t, I_ext):
        self.gpu_simulate_init(t,I_ext)
        self.cuda_gpu_simulate(np.int32(self.n),np.float32(self.dt),
                          self.g_x0.gpudata, self.g_x.gpudata, self.g_I.gpudata,
                          block=(4,1,1), grid=(1,1))
        self.gpu_simulate_finish()

if __name__ == '__main__':
    dur_t = 1.0
    dt    = 1e-5
    t = np.arange(0, dur_t, dt).astype(np.float32)

    I_ext = np.zeros_like(t).astype(np.float64)
    I_ext[ slice(int(np.ceil(0.25/dt)),np.ceil(0.75/dt)) ] = 10.

    hhn = Hodgkin_Huxley_Neuron()
    #hhn.cpu_simulate(t,I_ext)
    hhn.gpu_simulate(t,I_ext)

    # Generate figures on the same chart
    p.clf()
    p.subplot(221);p.plot(t, hhn.V)
    p.title('Membrane Voltage')
    p.xlabel(r'time (sec)');p.ylabel(r'V (mV)')
    p.subplot(222);p.plot(t, hhn.N)
    p.title('State Variable: n')
    p.xlabel(r'time (sec)');p.ylabel(r'n')
    p.subplot(223);p.plot(t, hhn.M)
    p.title('State Variable: m')
    p.xlabel(r'time (sec)');p.ylabel(r'm')
    p.subplot(224);p.plot(t, hhn.H)
    p.title('State Variable: h')
    p.xlabel(r'time (sec)');p.ylabel(r'h')
    p.tight_layout() # prevent subplots from overlapping
    p.savefig('Vmnh.png')

    p.clf()
    p.plot(hhn.N, hhn.V)
    p.title('Hodgkin-Huxley Phase Response Curve')
    p.ylabel(r'V (mV)')
    p.xlabel(r'n')
    p.savefig('nvsV.png')
