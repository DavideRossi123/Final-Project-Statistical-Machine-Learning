# Final-Project-Statistical-Machine-Learning

questions:
- we decided to focus on POM 
- simulation / real dataset, what in the report?
- fine to generate y assuming underlying latent variable and how we generated thresholds
- parameters to change: sample size, number of classes of y, set of thresholds, magnitude of beta
- shall compare probit or can avoid
- in the presentation/report are we expected to explain properly the mathematical explanation of the model or directly focus on our study


STEPS: 

SIMULATION:

compare:
- multinomial
- logit
- probit to decide 
- logit with splines / more complex relationships
- logit with regularised parameters: lasso, ridge, elastic net with mixing parameter 0.5

to vary potentially:
- sample size
- number of classes
- set of threshold
- number of covariates and type of relationship (useful for gams)
- magnitude of beta (set to some to 0 to see performance regularised models)
- how they perform in case where proportional odds fails


assessment of performance:
- MAE, MZE: do with k-fold cv
- biasedness and sd of beta over 1000? repetitions












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
- logit with regularised parameters: k-fold to choose lambda.

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
