# Introduction
This repository is supplementary to the paper "Extending Drift Detection Methods to Identify When Exactly the Change Happened" by Markus Vieth, Alexander Schulz, and Barbara Hammer, presented at the International Work-Conference on Artificial Neural Networks 2023.

# Abstract
Data changing, or drifting, over time is a major problem when using classical machine learning on data streams. One approach to deal with this is to detect changes and react accordingly, for example by retraining the model.
Most existing drift detection methods only report that a drift has happened between two time windows, but not when exactly. In this paper, we present extensions for three popular methods, MMDDDM, HDDDM, and D3, to determine precisely when the drift happened, i.e.\ between which samples. One major advantage of our extensions is that no additional hyperparameters are required.
In experiments, with an emphasis on high-dimensional, real-world datasets, we show that they successfully identify when the drifts happen, and in some cases even lead to fewer false positives and false negatives (undetected drifts), while making the methods only negligibly slower.
In general, our extensions may enable a faster, more robust adaptation to changes in data streams.

# Overview
The files `d3.py`, `MMDDDM.py`, and `HDDDM.py` contain the implementations of the corresponding drift detection methods. `datasets.py` contains functions to generate the datasets used in the paper's experiments. 

## Executables
`show.py` visualizes the results.  
`run_experiments.py` can be used to run the experiments from the paper. The following command line arguments are possible for `run_experiments.py`:
- `D3`, `MMDDDM`, `HDDDM`, and `covtype`, `rialto`, `mnist`, `music` run the experiments for the corresponding method and dataset, respectively. All combinations of specified methods and specified datasets are run
- `visualize` and `verbose` tell the methods to do additional visualizations and command line output for each drift. These are useful for debugging or understanding the methods
- `serial` makes the experiments run in serial, default is parallel (multiple processes)
- `stride1` tells the methods (if implemented in the method) to perform the drift test for every incoming sample, instead of every few samples as the methods usually do
- `nolocalizedrifts` tells the methods to not compute when exactly the drifts happened (the main idea of the paper)
- `<number>runs` says to perform `number` runs of each method-parameter-dataset combination (each run with a different random seed). For example, `200runs` would do this 200 times. Default is 250