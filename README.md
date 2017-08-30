# Abstract

This project builds a network performance prediction system that can learn from 
historical observed network metrics and predict network performance.

The project can be generalized to predict other network metrics as well, such
as network uptime or anomalies of any kind, as long as you have sufficient
relevant data samples. 

Data doesn't lie, but we must know the data first :)

# Getting Started

The project has 4 different implementations: 

1. [in C++ based on Neural Net of Intel DAAL library and MPICH](https://github.com/fleapapa/network_performance_prediction/tree/master/daal)
2. [in Python based on DNN of Tensorflow](https://github.com/fleapapa/network_performance_prediction/tree/master/tensorflow)
3. [in Python based on regressors of scipy-sklearn](https://github.com/fleapapa/network_performance_prediction/tree/master/sklearn)
4. [in Python based on regressors of Spark ML](https://github.com/fleapapa/network_performance_prediction/tree/master/spark)


# Requirements

Requirements for each of the 4 implementations are:
1. GNU C++11, Intel DAAL library, MPICH
2. Python 2.7, Tensorflow, Jupyter Notebook
3. Python 2.7, scipy-sklearn, Jupyter Notebook
4. Spark ML, Spark SQL, Jupyter Notebook

This project uses the same data samples as in [network_profile_recommender project](https://github.com/fleapapa/network_profile_recommender). 

# Contact

fleapapa@gmail.com


# License

This project is released under the MIT License.
