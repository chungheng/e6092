# Recitation 3 #

## LPU and BaseNeuron Class ##

`NeuroKernel` is designed for realizing massively parallel simulation of
multiple neural networks on multiple GPU instances, and it provides the neural
network designers with a common interface for crosstalk among networks across
different GPU instances. However, `NeuroKernel` does not put constraint nor
provide a framework on how a neural network is designed. `LPU` class comes in
to solve this issue, and its features can be summarized as follows,

* be controlled by NeuroKernel manager.
* parse network configurations into Python data type.
* evoke GPU kernel calls of neurons and synapse.

A `LPU` can contain multiple neuron types and synapse types. In each iteration
during runtime, a `LPU` will go through following steps: (i) read in, if any,
input stimuli to neurons; (ii) for each neuron types, performing on GPU,
compute the synaptic current, and then update the state variables of all
neurons of the same type; (iii) for each synapse types, performing on GPU,
update the state variables of all synapses of the same type. We excerpt a chunk
of source code from `LPU` class, and put it below to highlight the three steps,

    class LPU(Module, object):
        def __init__(...):
        ...
        def run_step(...):
            ...
            if self.input_file is not None:
                self._read_external_input()  # ex, stimuli to neurons
            ...
            for neuron in self.neurons:
                neuron.update_I(...)         # compute neuron synaptic current
                neuron.eval()                # update neuron state variables
            ...
            for synapse in self.synapses:
                synapse.update_state(...)    # update synapse state variables
            ...

Note that `neuron.update_I()`, `neuron.eval()`, and `synapse.update_state()`
are actually a wrapper to GPU kernel calls. `neuron`'s and `synapse`'s above
should be subclasses of `BaseNeuron` and `BaseSynapse`, respectively.
`BaseNeuron` or its subclass contains the GPU kernels with their associated
wrappers, and handles the memory allocation on GPU. To have a better sense of
what `BaseNeuron` does, we list the skeleton code of a neuron class named
`MyNeuron` inheriting from `BaseNeuron` below,

    class MyNeuron(BaseNeuron):

        def __init__(...):
            ...
            self._gpu_func = self._get_gpu_func

        @abstractmethod
        def eval(self):
            # where you call GPU kernel
            self._gpu_fun(...)

        def _get_gpu_func(...):
            cuda_src = """
                    # define DEFINE_A AS_B
                    ...

                    __global__  void myneuron(void* argv1, void* argv2, ...)
                    {
                        ....
                    }
                    """
            # use PyCuda to call NVCC to compile cuda code to python function
            mod = SourceModule(cuda_src, options = ["--ptxas-options=-v"])
            func = mod.get_function("myneuron")
            ....
            return func

During the initialization, `_get_gpu_func` is called to use `PyCuda` to compile
`CUDA` code into a Python function, and returns the resultant Python function
into an object named `_gpu_fun`. Later, during the runtime when `LPU` calls
`MyNeuron.eval()`, the GPU func `_gpu_fun` will then be executed. You may
wonder why we need to provide double wrapper of the GPU function, namely
`eval()` and `_gpu_fun()`. The reason is two fold: (i) `eval()` is inherited
from `BaseNeuron`, and it has predefined input argument lists; (ii) You may
wish to do some extra things in `eval()`, such as logging or providing handles
for debugging.

## Circuit Configuration ##

Currently, `LPU` supports neural circuit configuration in the
[GEXF](http://gexf.net/format/) (Graph Exchange XML Format) format. *GEXF* is
a language for specifying complex networks structures, their associated data and
dynamics. A file in *GEXF* format is usually simple, human-readable, and self
defined. In general, a *GEXF* file contains two main parts: (i) attribute
definitions for nodes and edges; (ii) data of nodes and synapse. What
follows is taken from the
[Lamina](https://github.com/neurokernel/neurokernel/blob/master/examples/vision/data/lamina.gexf.gz)
configuration in the vision examples,

     1 <?xml version="1.0" encoding="utf-8"?>
     2 <gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">
     3 <graph defaultedgetype="directed">
     4    <attributes class="node">
     5        <attribute id="0" title="model" type="string"/>
     6        ...
     7        <attribute id="20" title="b" type="float"/>
     8    </attributes>
     9    <attributes class="edge">
    10        <attribute id="0" title="reverse" type="float"/>
    11        ...
    12        <attribute id="8" title="conductance" type="boolean"/>
    13    </attributes>
    14    <nodes>
    15      <node id="0">
    16          <attvalues>
    17            <attvalue for="0" value="MorrisLecar"/>
    18            ....
    19            <attvalue for="14" value="1"/>
    20            <attvalue for="15" value="false"/>
    21         </attvalues>
    22       </node>
    23       ...
    23    </nodes>
    24    ...

Similar to most *XML* format, a scope in *GEXF* file starts with `<xxx>` and
ends with `</xxx>`. Line 4 to 8 is the declaration of attributes of node. Each
node attribute is associated with an *ID*, and this *ID* will be used when we
define a particular node. For example, in line 19,

    17            <attvalue for="0" value="MorrisLecar"/>

we the value of the attribute 0, which is equivalent to _"model"_ attribute,
to "MorrisLecar" for node 0.
