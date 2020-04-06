# LiNGAM - Discovery of non-gaussian linear causal models

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/cdt15/lingam/blob/master/LICENSE)
[![Read the Docs](https://readthedocs.org/projects/lingam/badge/?version=latest)](https://lingam.readthedocs.io/)

LiNGAM is a new method for estimating structural equation models or linear Bayesian networks. It is based on using the non-Gaussianity of the data.

* [The LiNGAM Project](https://sites.google.com/site/sshimizu06/lingam)

## Requirements
* Python3
* numpy
* scipy
* scikit-learn
* graphviz
* statsmodels

## Installation
To install lingam package, use `pip` as follows:

```
$ pip install lingam
```

## Documentation
[Tutrial and API reference](https://lingam.readthedocs.io/)

## License
This project is licensed under the terms of the [MIT license](./LICENSE).

## References
Should you use this package for performing **ICA-based LiNGAM algorithm**, we kindly
request you to cite the following paper:
* S. Shimizu, P. O. Hoyer, A. Hyvärinen and A. Kerminen. **A linear non-gaussian acyclic model for causal discovery**. *Journal of Machine Learning Research*, 7: 2003--2030, 2006. [[PDF]](http://www.jmlr.org/papers/volume7/shimizu06a/shimizu06a.pdf)

Should you use this package for performing **DirectLiNGAM algorithm**, we kindly
request you to cite the following two papers:
* S. Shimizu, T. Inazumi, Y. Sogawa, A. Hyvärinen, Y. Kawahara, T. Washio, P. O. Hoyer and K. Bollen. **DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model**. *Journal of Machine Learning Research*, 12(Apr): 1225--1248, 2011. [[PDF]](http://www.jmlr.org/papers/volume12/shimizu11a/shimizu11a.pdf)
* A. Hyvärinen and S. M. Smith. **Pairwise likelihood ratios for estimation of non-Gaussian structural equation models**. *Journal of Machine Learning Research*, 14(Jan): 111--152, 2013. [[PDF]](http://www.jmlr.org/papers/volume14/hyvarinen13a/hyvarinen13a.pdf)

Should you use this package for performing **LiNGAM for multiple
groups**, we kindly request you to cite the following paper:
* S. Shimizu. **Joint estimation of linear non-Gaussian acyclic models**. *Neurocomputing*, 81: 104-107, 2012. [[PDF]](http://dx.doi.org/10.1016/j.neucom.2011.11.005)

Should you use this package for performing **VAR-LiNGAM**, we kindly request you to cite the following paper:
* A. Hyvärinen, K. Zhang, S. Shimizu, and P. O. Hoyer. **Estimation of a structural vector autoregression model using non-Gaussianity**. *Journal of Machine Learning Research*, 11: 1709-1731, 2010. [[PDF]](http://www.jmlr.org/papers/volume11/hyvarinen10a/hyvarinen10a.pdf)

Should you use this package for performing **VARMA-LiNGAM**, we kindly request you to cite the following paper:
* Y. Kawahara, S. Shimizu and T. Washio. **Analyzing relationships among ARMA processes based on non-Gaussianity of external influences**. *Neurocomputing*, 74(12-13): 2212-2221, 2011. [[PDF]](http://dx.doi.org/10.1016/j.neucom.2011.02.008)

Should you use this package for performing **estimation of intervension effects on prediction**, we kindly request you to cite the following paper:
* P. Blöbaum and S. Shimizu. **Estimation of interventional effects of features on prediction**. In Proc. 2017 IEEE International Workshop on Machine Learning for Signal Processing (MLSP2017), pp. 1--6, Tokyo, Japan, 2017. [[PDF]](https://arxiv.org/abs/1709.00776)
