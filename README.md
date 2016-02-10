# Neuronal ShaderToy Framework

Experiments with multi-layer neural networks on http://shadertoy.com

### status

- classic multilayer perceptron with states, weights, and errors in only two render buffers 
- basically working for up to 4 layers (2 hidden)
- max number of layers is fixed in code but can be expanded. The lack of const arrays in webgl requires a lot of workaround. Also limited by screen resolution
- time inefficient: 4 layers need 8 frames for one training case: only 7.5 updates per second. GPUs could do *much* better
- hard to analyze performance because of slowness..
- can save two frames by not copying input state and pre-calcing error, which would make code less coherent for layers
