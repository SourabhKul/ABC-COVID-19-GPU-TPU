# ABC-COVID-19-GPU
This repo contains the GPU code of parallelized Approximate-Bayesian-Computation(ABC) simulation-based inference for a stochastic epidemiology model for COVID-19. The model and parallelized ABC algorithm are described in the following publication:

 "Accelerating Simulation-based Inference with Emerging AI Hardware", S Kulkarni, A Tsyplikhin, MM Krell, and CA Moritz, IEEE International Conference on Rebooting Computing (ICRC), 2020.
  
The data is obtained from [JHU CSSE COVID-19](https://github.com/CSSEGISandData/COVID-19).
It contains the COVID-19 case data (confirmed active cases, 
confirmed recovered cases, and confirmed deaths) for all countries. In this example we utilize case data of Italy.

There is an accompanying repo in [Graphcore Demos](https://github.com/graphcore/demos/tree/master/tensorflow2/ABC_COVID-19) in which the same model is implemented in Graphcore MK1 IPUs as part of the comparative analysis performed in the paper.
