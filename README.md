# influential <a href='https://github.com/asalavaty/python-influential'><img src='https://raw.githubusercontent.com/asalavaty/influential/master/man/figures/Symbol.png' align="right" height="221" /></a>

<!-- badges: start -->

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/influential)](https://pypi.python.org/pypi/influential)
[![PyPI wheels](https://img.shields.io/pypi/wheel/influential.svg)](https://pypi.python.org/pypi/influential)
[![](https://img.shields.io/badge/Integrated%20Value%20of%20Influence-IVI-blue.svg)](https://doi.org/10.1016/j.patter.2020.100052)
[![](https://img.shields.io/badge/SIR--based%20Influence%20Ranking-SIRIR-green.svg)](https://doi.org/10.1016/j.patter.2020.100052)
[![](https://img.shields.io/badge/Experimental%20data--based%20Integrative%20Ranking-ExIR-blue.svg)](https://www.biorxiv.org/content/10.1101/2022.10.03.510585v1.abstract)
<!-- badges: end -->

## Overview

The goal of `influential` is to help identification of the most
`influential` nodes in a network as well as the classification and
ranking of top candidate features. This package contains functions for
the classification and ranking of features, reconstruction of networks
from adjacency matrices and data frames, analysis of the topology of the
network and calculation of centrality measures as well as a novel and
powerful `influential` node ranking. The **Experimental data-based
Integrative Ranking (ExIR)** is a sophisticated model for classification
and ranking of the top candidate features based on only the experimental
data. The first integrative method, namely the **Integrated Value of
Influence (IVI)**, that captures all topological dimensions of the
network for the identification of network most `influential` nodes is
also provided as a function. Also, neighborhood connectivity, H-index,
local H-index, and collective influence (CI), all of which required
centrality measures for the calculation of **IVI**, are for the first
time provided in a python package. Additionally, a function is provided for
running **SIRIR** model, which is the combination of leave-one-out cross
validation technique and the conventional SIR model, on a network to
unsupervisedly rank the true influence of vertices.

Check out [**our paper**](https://doi.org/10.1016/j.patter.2020.100052)
for a more complete description of the IVI formula and all of its
underpinning methods and analyses.

Also, read our [**preprint**](https://www.biorxiv.org/content/10.1101/2022.10.03.510585v1.abstract) on the ExIR model and 
its validations.

## Author

The `influential` package was written by [Abbas (Adrian)
Salavaty](https://asalavaty.com/)

## Advisors

Mirana Ramialison and Peter D. Currie

## How to Install

You can install the official [PyPI](http://pypi.python.org/pypi/influential/) of the
`influential` with the following code:

``` python
pip install influential
```

Or the development version from GitHub:

``` python
pip install git+https://github.com/asalavaty/python-influential.git#egg=influential
```

**Note**: If you are using Python 3 you may need to use `pip3` instead of `pip`, as follows.

``` python
pip3 install influential
pip3 install git+https://github.com/asalavaty/python-influential.git#egg=influential
```

## Shiny apps

- [Influential Software Package web
  portal](https://influential.erc.monash.edu/)

- [IVI Shiny App](https://influential.erc.monash.edu/IVI/): A shiny app
  for the calculation of the Integrated Value of Influence (IVI) of
  network nodes as well as IVI-based visualization of the network.

- [ExIR Shiny App](https://influential.erc.monash.edu/ExIR/): A shiny
  app for running the Experimental-data-based Integrative Ranking (ExIR)
  model as well as visualization of its results.


## How to cite `influential`

To cite `influential`, please cite its associated paper:

- Integrated Value of Influence: An Integrative Method for the
  Identification of the Most Influential Nodes within Networks. Abbas
  Salavaty, Mirana Ramialison, Peter D Currie. *Patterns*. 2020.08.14
  ([Read online](https://doi.org/10.1016/j.patter.2020.100052)).

## How to contribute

Please don’t hesitate to report any bugs/issues and request for
enhancement or any other contributions. To submit a bug report or
enhancement request, please use the [`python-influential` GitHub issues
tracker](https://github.com/asalavaty/python-influential/issues).
