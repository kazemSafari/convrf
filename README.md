# convrf

Implementation of Directional Receptive Field convolutional layer in pytorch. This idea is taken from "Structured Receptive Fields" (https://arxiv.org/abs/1605.02971) paper. To run the code:


0. add your own 3x3, 5x5, and 7x7 filterbanks to directional_filterbanks/2d/ folder. They should be named following the convention "3x3.npy", "5x5.npy", and "7x7.npy"
1. download and install using pip install -e convrf
2. Take a look at example.py for an example of convrf convolutional layer usage.
3. play around with KERNELS_RATIO parameter to see what value fits best to your application. KERNELS_RATIO = 1 uses all the filters in the family.
