# LiNGAM - Discovery of non-gaussian linear causal models

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/cdt15/lingam/blob/master/LICENSE)
[![Read the Docs](https://readthedocs.org/projects/lingam/badge/?version=latest)](https://lingam.readthedocs.io/)

LiNGAM is a new method for estimating structural equation models or linear Bayesian networks. It is based on using the non-Gaussianity of the data.

* [The LiNGAM Project](https://sites.google.com/view/sshimizu06/lingam)

## Requirements

* Python3

* numpy

* scipy

* scikit-learn

* graphviz

* statsmodels

- networkx

- pandas

- itertools

- semopy


## Installation

To install lingam package, use `pip` as follows:

```sh
pip install lingam
```

## Usage

```python
import numpy as np
import pandas as pd
import lingam

# To run causal discovery, we create a DirectLiNGAM object and call the fit method.
model = lingam.DirectLiNGAM()
model.fit(X)

# Using the causal_order_ properties,
# we can see the causal ordering as a result of the causal discovery.
print(model.causal_order_)

# Also, using the adjacency_matrix_ properties,
# we can see the adjacency matrix as a result of the causal discovery.
print(model.adjacency_matrix_)
```

## Documentation

[Tutorial and API reference](https://lingam.readthedocs.io/)

[Tutorial slides](https://speakerdeck.com/sshimizu2006/lingam-python-package)

## Examples

We provide several examples of running the LiNGAM algorithm in Jupyter Notebook.
 [lingam/examples](./examples)

## License

This project is licensed under the terms of the [MIT license](./LICENSE).

## Contribution

For guidelines how to contribute to lingam package, take a look at [CONTRIBUTING.md](./CONTRIBUTING.md).

## References

### Package

If you find our package useful, please cite the following paper:

* T. Ikeuchi, M. Ide, Y. Zeng, T. N. Maeda, and S. Shimizu. **Python package for causal discovery based on LiNGAM**. *Journal of Machine Learning Research*, 24(14): 1−8, 2023. [[PDF]](https://jmlr.org/papers/v24/21-0321.html)


### Basic DAG model

Should you use this package for performing **ICA-based LiNGAM algorithm**, we kindly request you to cite the following paper:

* S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. Kerminen. **A linear non-gaussian acyclic model for causal discovery**. *Journal of Machine Learning Research*, 7: 2003--2030, 2006. [[PDF]](http://www.jmlr.org/papers/volume7/shimizu06a/shimizu06a.pdf)

Should you use this package for performing **DirectLiNGAM algorithm**, we kindly request you to cite the following two papers:

* S. Shimizu, T. Inazumi, Y. Sogawa, A. Hyvärinen, Y. Kawahara, T. Washio, P. O. Hoyer and K. Bollen. **DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model**. *Journal of Machine Learning Research*, 12(Apr): 1225--1248, 2011. [[PDF]](http://www.jmlr.org/papers/volume12/shimizu11a/shimizu11a.pdf)
* A. Hyvärinen and S. M. Smith. **Pairwise likelihood ratios for estimation of non-Gaussian structural equation models**. *Journal of Machine Learning Research*, 14(Jan): 111--152, 2013. [[PDF]](http://www.jmlr.org/papers/volume14/hyvarinen13a/hyvarinen13a.pdf)

Should you use this package for performing **RESIT algorithm**, we kindly request you to cite the following paper:

* J. Peters, J. M. Mooij, D. Janzing, and B. Schölkopf. **Causal Discovery with Continuous Additive Noise Models**. *Journal of Machine Learning Research*, 15(58): 2009--2053, 2014. [[PDF]](http://www.jmlr.org/papers/volume15/peters14a/peters14a.pdf)

Should you use this package for performing **GroupDirectLiNGAM algorithm**, we kindly request you to cite the following paper:

* D. Entner and P. O. Hoyer. **Estimating a causal order among groups of variables in linear models.** In Proc. 22nd International Conference on Artificial Neural Networks (ICANN2012), pp. 83--90, Lausanne, Switzerland, 2012. [[PDF]](https://link.springer.com/chapter/10.1007/978-3-642-33266-1_11)


### Time series

Should you use this package for performing **VAR-LiNGAM**, we kindly request you to cite the following paper:

* A. Hyvärinen, K. Zhang, S. Shimizu, and P. O. Hoyer. **Estimation of a structural vector autoregression model using non-Gaussianity**. *Journal of Machine Learning Research*, 11: 1709-1731, 2010. [[PDF]](http://www.jmlr.org/papers/volume11/hyvarinen10a/hyvarinen10a.pdf)

Should you use this package for performing **VARMA-LiNGAM**, we kindly request you to cite the following paper:

* Y. Kawahara, S. Shimizu and T. Washio. **Analyzing relationships among ARMA processes based on non-Gaussianity of external influences**. *Neurocomputing*, 74(12-13): 2212-2221, 2011. [[PDF]](http://dx.doi.org/10.1016/j.neucom.2011.02.008)


### Multiple datasets

Should you use this package for performing **DirectLiNGAM for multiple groups**, we kindly request you to cite the following paper:

* S. Shimizu. **Joint estimation of linear non-Gaussian acyclic models**. *Neurocomputing*, 81: 104-107, 2012. [[PDF]](http://dx.doi.org/10.1016/j.neucom.2011.11.005)

Should you use this package for performing **LiNGAM for longitudinal data**, we kindly request you to cite the following paper:

* K. Kadowaki, S. Shimizu, and T. Washio. **Estimation of causal structures in longitudinal data using non-Gaussianity**. In Proc. 23rd IEEE International Workshop on Machine Learning for Signal Processing (MLSP2013), pp. 1--6, Southampton, United Kingdom, 2013. [[PDF]](https://doi.org/10.1109/MLSP.2013.6661912)


### Latent confounders and latent factors

Should you use this package for performing **BottomUpParceLiNGAM** with Algorithm 1 of the paper below except Step 2 for estimating causal orders, we kindly request you to cite the following paper:

* T. Tashiro, S. Shimizu, A. Hyvärinen, T. Washio. **ParceLiNGAM: a causal ordering method robust against latent confounders**. Neural computation, 26(1): 57-83, 2014. [[PDF]](https://ieeexplore.ieee.org/abstract/document/6797648)

Should you use this package for performing **RCD algorithm**, we kindly request you to cite the following paper:

* T. N. Maeda and S. Shimizu. **RCD: Repetitive causal discovery of linear non-Gaussian acyclic models with latent confounders.** In Proc. 23rd International Conference on Artificial Intelligence and Statistics (AISTATS2020), Palermo, Sicily, Italy. PMLR  108:735-745, 2020. [[PDF]](http://proceedings.mlr.press/v108/maeda20a.html)

Should you use this package for performing **LiNA algorithm**, we kindly request you to cite the following paper:

* Y. Zeng, S. Shimizu, R. Cai, F. Xie, M. Yamamoto and Z. Hao. **Causal Discovery with Multi-Domain LiNGAM for Latent Factors**. In Proc. of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21), 2021: 2097--2103. [[PDF](https://www.ijcai.org/proceedings/2021/289)]

Should you use this package for performing **CAM-UV algorithm**, we kindly request you to cite the following paper:

* T. N. Maeda and S. Shimizu. **Causal additive models with unobserved variables.** In Proc. 37th Conference on Uncertainty in Artificial Intelligence (UAI). PMLR 161:97-106, 2021. [[PDF]](https://proceedings.mlr.press/v161/maeda21a.html)

Should you use this package for performing **GroupLiNGAM algorithm**, we kindly request you to cite the following paper:

* Y. Kawahara, K. Bollen, S. Shimizu and T. Washio. **GroupLiNGAM: Linear non-Gaussian acyclic models for sets of variables.** Arxiv preprint arXiv:1006.5041, 2010. [[PDF]](https://arxiv.org/abs/1006.5041)


### Causality and prediction

Should you use this package for performing **estimation of intervention effects on prediction**, we kindly request you to cite the following paper:

* P. Blöbaum and S. Shimizu. **Estimation of interventional effects of features on prediction**. In Proc. 2017 IEEE International Workshop on Machine Learning for Signal Processing (MLSP2017), pp. 1--6, Tokyo, Japan, 2017. [[PDF]](https://doi.org/10.1109/MLSP.2017.8168175)

### Mixed data

Should you use this package for performing **LiM algorithm**, we kindly request you to cite the following paper:

* Y. Zeng, S. Shimizu, H. Matsui, F. Sun. **Causal discovery for linear mixed data**. In Proc. First Conference on Causal Learning and Reasoning (CLeaR2022). PMLR 177, pp. 994-1009, 2022. [[PDF]](https://proceedings.mlr.press/v177/zeng22a.html)


