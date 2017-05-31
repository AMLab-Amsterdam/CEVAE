# CEVAE
This repository contains the code for the  Causal Effect Variational 
Autoencoder (CEVAE) model as developed at [1]. 

Sample experiment
---
To perform a sample run of CEVAE on 10 replications of the Infant Health
 and Development Program (IHDP) dataset just type:
`python cevae_ihdp.py`
 
Other datasets
---
To employ CEVAE for other datasets you can just mimic the structure of the 
IHDP class at `datasets.py`. Do note that you will also have to specify 
appropriate distributions via Edward for the covariates at `x`, treatments at `t` and outcomes at `y`. For example, poisson for covariates which are counts, or categorical/Bernoulli for discrete outcomes. 
 
The definition of the distribution type for the treatment type and outcome
can be easily changed by modifying lines 93, 99 for the generative model and 
 by modifying lines 104 and 109 for the inference model at `cevae_ihdp.py`.
 
Also note that IHDP, being a synthetic dataset, has both the treated and control conditional means (mu1 and mu0) and the factual and counterfactual outcomes (y and y_cf). These are used in evalution.py to calculate various performance metrics. For a dataset without the counterfactuals you will have to avoid calling these evaluation functions and instead write your own evaluation procedure.
 
 
Requirements
---
- Edward 1.3.1
- Tensorflow 1.1.0
- Progressbar 2.3
- Scikit-learn 0.18.1

References
---
[1] [Causal Effect Inference with Deep Latent-Variable Models](https://arxiv.org/abs/1705.08821)
Christos Louizos, Uri Shalit, Joris Mooij, David Sontag, Richard Zemel, Max Welling, 2017


