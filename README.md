# Normalized Activations

**NOTE**: it turns out similar ideas have already recently been investigated:
- arxiv: [2101.08692](https://arxiv.org/abs/2101.08692)
- arxiv: [2102.06171](https://arxiv.org/abs/2102.06171)

Experiments based on normalising neural network activations (or weights)
in a pretraining step, using random input noise.

The network is trained to achieve a target mean and standard deviation after each layer,
based on this normally distributed noise.

Results show that this method significantly improves training speed and
performance using the same number of weights.

![](docs/img/21_layer_ae_swish_marked.png)

Generally the mean should be zero after each layer, and the standard deviation should
slowly increase from very little `< 0.1` to a larger value `> 0.5`.

- Further experiments are needed to figure out the exact mechanisms at play.
