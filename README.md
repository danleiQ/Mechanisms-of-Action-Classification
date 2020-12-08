# Mechanisms-of-Action-prediction
Multi-Label Classification


This project is based on the Kaggle Competition- Mechanisms of Action (MoA) Prediction,with the goal to “classify drugs based on their biological activity”. This is a multi-label classification task. Drugs can have multiple MoA annotations which describe binary responses from different cell types in different ways. 

You can find the homepage and datasets of the competition from the following link:
https://www.kaggle.com/c/lish-moa/overview

# Result
| Single Model | Seeds | K-folds | Cross Validation without Drug_id | Cross Validation with Drug_id | Public Score | Private Score | 
| ----- | ----- | ----- | ----- | ----- | ----- |  ----- | 
| Tabnet | 1 |10 | 0.016717 |  |0.01841| 0.01632 |
| 2-Phase NN With Transfer Learning | 7 | 7 | |  0.01563 |0.01833| 0.01623 |
|2-Heads Resnet NN | 7 |10 |0.01656 |   |0.01850| 0.01635 |
|Ensemble with average weights |  | | |   |0.01824| 0.01609 |

# References
Here are some resources I've been learning from:

[Model Achitecture]:

TabNet:

Paper: https://arxiv.org/abs/1908.07442

Pytorch Implementation: https://github.com/dreamquark-ai/tabnet

Tensorflow 2.0 Implementation: https://www.kaggle.com/marcusgawronsky/tabnet-in-tensorflow-2-0

[Feature Engineering]:

Permutation Importance:

Introduction:https://www.kaggle.com/dansbecker/permutation-importance

eli5 implementation: https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html

sklearn implementation: https://scikit-learn.org/stable/modules/permutation_importance.html

[Hyperparameters Tuning]:

Optuna: 

Tutorial: https://www.kaggle.com/fanvacoolt/tutorial-on-hyperopt

Optuna: https://optuna.readthedocs.io/en/v2.1.0/reference/generated/optuna.visualization.plot_intermediate_values.html

[Transfer Learning]:

https://www.kaggle.com/chriscc/kubi-pytorch-moa-transfer

https://www.kaggle.com/thehemen/pytorch-transfer-learning-with-k-folds-by-drug-ids

K-means:
https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1

https://www.kaggle.com/yerramvarun/deciding-clusters-in-kmeans-silhouette-coeff

https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c

PCA 
https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/

https://www.kaggle.com/kushal1506/deciding-n-components-in-pca

T-test

https://www.kaggle.com/demetrypascal/t-test-pca-rfe-logistic-regression#Select-only-important

Adversarial Validation

https://towardsdatascience.com/adversarial-validation-ca69303543cd

Label Smoothing

https://leimao.github.io/blog/Label-Smoothing

