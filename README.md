# aligner_gpu_version
Author: Enliang Li

mian translation for the forward algorithm (gpu-version)

Latest Version: 1.2 on Apr.15th 2019

# Use following command to build:
using the "compile.sh" to build the sources 

# Performance Details (TITAN Xp) [HOLMES flow (27 states)]
## forward_matrix [x_dim+1][y_dim+1][batch][states-1]

forward_matrix: torch.Size

([6, 7, 1, 26])

10.79748 ms / call (float)

11.009 ms / call (double)

forward_matrix: torch.Size

([16, 19, 1, 26])

11.50272 ms / call (float)

11.9557 ms / call (double)

forward_matrix: torch.Size

([61, 127, 1, 26])

30.30934 ms / call (float)

33.30686 ms / call (double)


# Performance Details (TITAN Xp) [regular flow (3 states)]
## forward_matrix [x_dim+1][y_dim+1][batch][states-1]

forward_matrix: torch.Size

([16, 19, 1, 3])

12.6679 ms / call (float)

13.0121 ms / call (double)

forward_matrix: torch.Size

([16, 19, 3, 3])

12.9385 ms / call (float)

13.7454 ms / call (double)

forward_matrix: torch.Size

([159, 149, 1, 3])

69.3239 ms / call  (float)

72.4838 ms / call  (double)

# PyCUDA Support:
## copy-paste the function definition and call it with PyCUDA in your own flow
For examples, please refer to:

https://wiki.tiker.net/PyCuda/Examples

Copy Right Reserved by AndreasKloeckner
