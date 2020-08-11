# convrf

Implementation of Receptive Field convolutional layer in pytorch. This idea was originally presented in [Structured Receptive Fields](https://arxiv.org/abs/1605.02971) paper. To run the code:
 
1. download and install using pip install -e convrf
2. Take a look at example.py for an example of convrf convolutional layer usage.
3. Choose the fitlerbank: "nn_bank" or "frame" or "pframe". It is not recommended to use "nn_bank" as it almost always produces inferior results.
4. play around with "kernel_drop_rate" parameter to see what value fits best to your application. kernel_drop_rate = 0 uses all the filters in the family.
5. You can play around with "gain" parameter which enlarges or shrinks the support of the weight coefficients initialization kaiming uniform distribution if you feel like it, not it is not necessary.

In case you it in your research please cite these three papers: [Structured Receptive Fields](https://arxiv.org/abs/1605.02971), [On the design of multi-dimensional compactly supported parseval framelets with directional characteristics](https://www.sciencedirect.com/science/article/abs/pii/S0024379519303155), and the main paper which I will share the link shortly.
