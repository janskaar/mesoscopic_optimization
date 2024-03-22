 _________________________
< MESOSCOPIC OPTIMIZATION >
 -------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||

JAX implementation of the mesoscopic model described in [1], mainly used for investigating the expressiveness of the model by fitting it to population spiking histories of different neuron models. The log probability of any population spiking history can be computed, which allows for optimization by gradient ascent on the log likelihood. 

Example of a two-population network:
![example_fig_optimized_mesoscopic](https://github.com/janskaar/mesoscopic_optimization/assets/29370469/b6f5969b-ee6e-45a3-b188-7bc9a06d64a9)

