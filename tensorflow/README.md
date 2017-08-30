# Abstract

This directory contains a Python implementation of a network performance prediction 
application that learns from historical observed network metrics and predicts network 
performance.

The implementation is based on [Tensorflow DNNRegressor](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNRegressor). 

# Files

* [dnnr.ipynb](https://github.com/fleapapa/network_performance_prediction/blob/master/tensorflow/dnnr.ipynb) -  run w/o GPU.
* [dnnr_K80.ipynb](https://github.com/fleapapa/network_performance_prediction/blob/master/tensorflow/dnnr_K80.ipynb) - run with K80 GPU.

# Requirements

* [Tensorflow](https://www.tensorflow.org/install/install_linux)
* Python 2.7
* Jupyter Notebook

This project uses the same [data samples](https://github.com/fleapapa/network_profile_recommender/tree/master/data) as in [network_profile_recommender project](https://github.com/fleapapa/network_profile_recommender). 

# Contact

fleapapa@gmail.com


# License

This project is released under the MIT License.
