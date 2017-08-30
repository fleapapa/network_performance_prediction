# Abstract

This directory contains an C++ implementation of a network performance prediction 
application that learn from historical observed network metrics and predict network 
performance.

The implementation is based on Neural network algorithm of Intel DAAL library
and MPICH libary. It reuse DAAL sample code but adds quite some code for a
early evaluation stopper.

# Files

* [neural_net_dense_allgather_distributed_mpi2.cpp](https://github.com/fleapapa/network_performance_prediction/blob/master/daal/neural_net_dense_allgather_distributed_mpi2.cpp) -  application main code
* [neural_net_dense_allgather_distributed_mpi2.h](https://github.com/fleapapa/network_performance_prediction/blob/master/daal/neural_net_dense_allgather_distributed_mpi2.h) - neural net topology initialization code
* [neural_net_dense_allgather_distributed_mpi2.ipynb](https://github.com/fleapapa/network_performance_prediction/blob/master/daal/neural_net_dense_allgather_distributed_mpi2.ipynb) - prediction result visisualizer

# Getting Started

The application uses three hidden fully-connected layers and a softmax output layer.

The application iterates through a grid of hidden layer configs with different numbers
of neurons in the first two layers. The third layer has 101 neurons to make predictions
in the range of [0, 100].

See go.sh (or go2.sh) about how to prepare sample data and how to build/run the app.

# Requirements

* GNU C++11 
* Intel DAAL library
* MPICH library

To visualize the prediction results, it needs
* Python 2.7
* Jupyter Notebook

This project uses the same [data samples](https://github.com/fleapapa/network_profile_recommender/tree/master/data) as in [network_profile_recommender project](https://github.com/fleapapa/network_profile_recommender). 

# Contact

fleapapa@gmail.com


# License

This project is released under the MIT License.
