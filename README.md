# Final-Project-Statistical-Machine-Learning

questions:
- simulation / real dataset
- fine to generate y assuming underlying latent variable
- parameters to change: number of classes of y, set of thresholds, magnitude of beta


STEPS: 

SIMULATION:






APPLIED TO REAL DATASET:
setup:
response: house quality. it has 10 categories but merged into 7 (merged extremes). most obs in the middle categories
restrictred to house sell in 2009: n = 648
used 18 covariates, some with no effect to later test the efficiency of Lasso. checked correlations


k=10-fold analysis

models:
- multinomial
- logit / probit
- logit with gam
- regularised models: k-fold to choose lambda.

- to check performance in terms of MAE, MZE: k-fold (nested for regularised models)
- useful confusion matrix 




# Citazioni 
@article{gambarota2024ordinal,
  title={Ordinal regression models made easy: A tutorial on parameter interpretatiand added pvalues and ratio which is just exp(beta1) AND ratio_10 which is exp(beta1*10) and its the multiplicative change for going up 10 unitson, data simulation and power analysis},
  author={Gambarota, Filippo and Alto{\`e}, Gianmarco},
  journal={International Journal of Psychology},
  volume={59},
  number={6},
  pages={1263--1292},
  year={2024},
  publisher={Wiley Online Library}
}
