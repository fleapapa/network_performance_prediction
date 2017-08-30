/* file: neural_net_dense_asynch_distributed_mpi.cpp */
//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2017 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================

/*
!  Content:
!    C++ example of neural network training and scoring in the distributed processing mode
!    using asynchronous communications
!******************************************************************************/
//ppan+
#include <sys/types.h>
#include <unistd.h>
#include <thread>
#include <math.h>

#include <mpi.h>
#include "daal.h"
#include "service.h"
#include "neural_net_dense_distributed_mpi2.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::optimization_solver;
using namespace daal::algorithms::neural_networks;
using namespace daal::services;

typedef std::vector<byte> ByteBuffer;
typedef std::vector<MPI_Request> RequestBuffer;


const size_t nWorkers = 4;
const size_t nPartialResultsToUpdateWeights = nWorkers;

/* Input data set parameters */
//ppan+ see EC2 in /data/ml/daal/adnn
#define DDIR "../data/"
const string trainDatasetFileNames[nWorkers] =
{
    DDIR"trainx_xaa.csv",
    DDIR"trainx_xab.csv",
    DDIR"trainx_xac.csv",
    DDIR"trainx_xad.csv"
};
const string trainGroundTruthFileNames[nWorkers] =
{
    DDIR"trainy_xaa.csv",
    DDIR"trainy_xab.csv",
    DDIR"trainy_xac.csv",
    DDIR"trainy_xad.csv"
};
string testDatasetFile     = DDIR"testx.csv";
string testGroundTruthFile = DDIR"testy.csv";

const size_t batchSizeLocal = 20;
const size_t nEpoch = 1;

TensorPtr trainingData;
TensorPtr trainingGroundTruth;
prediction::ModelPtr predictionModel;
prediction::ResultPtr predictionResult;
training::TopologyPtr topology;
training::TopologyPtr topologyMaster;

/* Algorithms to train neural network */
training::Distributed<step1Local> netLocal;
training::Distributed<step2Master> netMaster;

LayerIds ids;

int rankId, comm_size;
#define mpi_root 0

static size_t ntrain;
static size_t epoch;

const int partialResultTag       = 0;
const int partialResultLengthTag = 1;
const int wbTag                  = 2;
const int tagNumberTrainSamples  = 7;	//ppan+
const int tagWorkerEopch         = 8;	//ppan+

void initializeNetwork();
void trainModel();
void testModel();
void printResults();

NumericTablePtr pullWeightsAndBiasesFromMaster(size_t &wbArchLength, ByteBuffer &wbBuffer, MPI_Request &wbRequest);

static void sendPartialResultToMaster(const training::PartialResultPtr &partialResult,
    size_t &partialResultArchLength,
    ByteBuffer &partialResultBuffer,
    MPI_Request &prRequest);

static training::PartialResultPtr deserializePartialResultFromNode(
    size_t &partialResultArchLength, byte *partialResultBuffer);

void wait_for_stop_training()
{
	if (mpi_root == rankId)
	{
		//master stops when all workers passes the last epoch
		while (::epoch < nEpoch)
		{
			for (int i = 0; i < nWorkers; i++)
			{
				size_t epoch;
				MPI_Request request;
				MPI_Irecv(&epoch, sizeof(size_t), MPI_BYTE, MPI_ANY_SOURCE, tagWorkerEopch, MPI_COMM_WORLD, &request);

				MPI_Status status;
				if (MPI_SUCCESS == MPI_Wait(&request, &status))
					std::cout << "worker " << status.MPI_SOURCE << " passes epoch " << epoch + 1 << " of " << nEpoch << std::endl;
			}
			std::cout << "all workers pass epoch " << ::epoch + 1 << " of " << nEpoch << std::endl;
			++::epoch;
		}
		std::cout << "passed all " << nEpoch << " epoches!" << std::endl;
	}
	else
	{
		//worker stops when master finds satisfied sum of derivative
		//TODO:
	}
}

int main(int argc, char *argv[])
{
	int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided != MPI_THREAD_MULTIPLE)
	{
		std::cerr << "MPI_THREAD_MULTIPLE not provided: " << provided << std::endl;
		exit(-1);
	}

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
	std::cout << "rank " << rankId << ", pid " << getpid() << std::endl;

	//ppan+ a thread to listen to a message to stop training
	std::thread(wait_for_stop_training).detach();

    initializeNetwork();

    trainModel();

    if(rankId == mpi_root)
    {
        testModel();
        printResults();
    }

    MPI_Finalize();

    return 0;
}

void initializeNetwork()
{
    /* Read training data set from a .csv file and create tensors to store input data */
    if (rankId == mpi_root)
    {
        trainingData = readTensorFromCSV(trainDatasetFileNames[0]);
        trainingGroundTruth = readTensorFromCSV(trainGroundTruthFileNames[0], true);
    }
    else
    {
        trainingData = readTensorFromCSV(trainDatasetFileNames[rankId - 1]);
        trainingGroundTruth = readTensorFromCSV(trainGroundTruthFileNames[rankId - 1], true);

		//ppan+ get the number of training samples at each worker
		ntrain = trainingData->getDimensions()[0];
    }

    Collection<size_t> dataDims = trainingData->getDimensions();

    /* Configure the neural network topology */
    topology = configureNet(&ids);

    training::ModelPtr trainingModel;
    if (rankId == mpi_root)
    {

		//ppan+ the master collects and sum up numbers of training samples from all worker nodes
		for (int i = 0; i < nWorkers; i++)
		{
			size_t numberTrainSamples;
			MPI_Request request;
			MPI_Irecv(&numberTrainSamples, sizeof(size_t), MPI_BYTE, MPI_ANY_SOURCE, tagNumberTrainSamples, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, MPI_STATUS_IGNORE);
			ntrain += numberTrainSamples;
		}
		std::cout << "total " << ntrain << " samples " << std::endl;

        /* Configure the neural network on master node */
        netMaster.parameter.batchSize = batchSizeLocal;

#if 0
        //this results in 3% accuracy with all NaN's in probability matrix
        SharedPtr<optimization_solver::lbfgs::Batch<float> > solver(new optimization_solver::lbfgs::Batch<float>());
#else
        /* Create AdaGrad optimization solver algorithm */
        SharedPtr<optimization_solver::adagrad::Batch<float> > solver(new optimization_solver::adagrad::Batch<float>());

        /* Set learning rate for the optimization solver used in the neural network */
        float learningRate = 0.001f;
        solver->parameter.learningRate = NumericTablePtr(new HomogenNumericTable<double>(1, 1, NumericTable::doAllocate, learningRate));
        solver->parameter.batchSize = 1;
        solver->parameter.optionalResultRequired = true;
#if 0	//not help
        solver->parameter.nIterations = 1 << 50;
        solver->parameter.accuracyThreshold = 1.0E-31;
#endif
#endif
        /* Set the optimization solver for the neural network training */
        netMaster.parameter.optimizationSolver = solver;

        /* Initialize the neural network on master node */
        netMaster.initialize(dataDims, *topology);

        trainingModel = netMaster.getResult()->get(training::model);
    }
    else
    {
		//ppan+ each worker tells the master how many samples it will train
		MPI_Send(&ntrain, sizeof(size_t), MPI_BYTE, mpi_root, tagNumberTrainSamples, MPI_COMM_WORLD);

        /* Configure the neural network on worker nodes */
        training::Distributed<step2Master> netInit;
        /* Set the batch size for the neural network training */
        netInit.parameter.batchSize = batchSizeLocal;
        netInit.initialize(dataDims, *topology);

        trainingModel = netInit.getResult()->get(training::model);

        netLocal.input.set(training::inputModel, trainingModel);

        /* Set the batch size for the neural network training */
        netLocal.parameter.batchSize = batchSizeLocal;
    }
}

void trainModel()
{
    ByteBuffer wbBuffer(0);             // buffer for serialized weights and biases
    size_t wbArchLength = 0;            // length of the buffer for serialized weights and biases

    ByteBuffer partialResultBuffer(0);  // buffer for serialized partial results (derivatives)
    size_t partialResultArchLength = 0; // length of the buffer for serialized partial results

    /* Run the neural network training on worker node */
    size_t nSamples = trainingData->getDimensionSize(0);

	//ppan+ somehow it crash if nSamples is not exact times of batchSizeLocal
	nSamples = nSamples / batchSizeLocal * batchSizeLocal;

    if (rankId == mpi_root)
    {
        /* Serialize weights and biases on master node */
        training::ModelPtr wbModel = netMaster.getPartialResult()->get(training::resultFromMaster)->get(training::model);
        checkPtr((void *)wbModel.get());
        NumericTablePtr wb = wbModel->getWeightsAndBiases();
        InputDataArchive wbDataArch;
        wb->serialize(wbDataArch);

        wbArchLength = wbDataArch.getSizeOfArchive();

        wbBuffer.resize(wbArchLength);
        wbDataArch.copyArchiveToArray(&wbBuffer[0], wbArchLength);
    }

    /* Broadcast the length of the buffer for serialized weights and biases */
    MPI_Bcast(&wbArchLength, sizeof(size_t), MPI_BYTE, mpi_root, MPI_COMM_WORLD);

    if (rankId != mpi_root)
    {
        /* Process input data on worker nodes */
        wbBuffer.resize(wbArchLength);

        MPI_Request prRequest;
        MPI_Request wbRequest;

		for (size_t ei = 0; ei < nEpoch; ei++)
		{
			for (size_t i = 0; i < nSamples; i += batchSizeLocal)
			{
				/* Compute weights and biases for the batch of inputs on worker nodes */

				/* Pass a training data set and dependent values to the algorithm */
				netLocal.input.set(training::data,        getNextSubtensor(trainingData,        i, batchSizeLocal));
				netLocal.input.set(training::groundTruth, getNextSubtensor(trainingGroundTruth, i, batchSizeLocal));

				if (i > 0 || ei > 0)	//ppan+	not the first batch
				{
					/* Pull the updated weights and biases from master node */
					NumericTablePtr wbLocal = pullWeightsAndBiasesFromMaster(wbArchLength, wbBuffer, wbRequest);
					netLocal.input.get(training::inputModel)->setWeightsAndBiases(wbLocal);
				}

				if (i + batchSizeLocal < nSamples || ei + 1 < nEpoch)	//ppan+ not the last batch
				{
					/* Request the updated weights and biases from master node */
					MPI_Irecv(&wbBuffer[0], wbArchLength, MPI_BYTE, mpi_root, wbTag, MPI_COMM_WORLD, &wbRequest);
				}

				/* Perform forward and backward pass through the neural network
				   to compute weights and biases derivatives on worker node */
				netLocal.compute();

#if 0
/* Get the model of the neural network on local node */
training::ModelPtr trainingModel = netLocal.input.get(training::inputModel);
/* Get the result of the loss layer */
services::SharedPtr<layers::forward::Result> lossLayerResult = trainingModel->getForwardLayer(ids.sm)->getLayerResult();
/* Print the value computed by the loss layer */
printf("xxx %d: %d, %d\n", rankId, ei, i);
printTensor(lossLayerResult->get(layers::forward::value), "loss value: ");
#endif

				/* Send the derivatives to master node */
				if (i > 0 || ei > 0)	//ppan+	not the first batch
				{
					MPI_Wait(&prRequest, MPI_STATUS_IGNORE);
				}

				sendPartialResultToMaster(netLocal.getPartialResult(), partialResultArchLength, partialResultBuffer, prRequest);

				if (0 == ei && 0 == i)	//ppan+ the first batch
				{
					MPI_Send(&partialResultArchLength, sizeof(size_t), MPI_BYTE, mpi_root,
								partialResultLengthTag, MPI_COMM_WORLD);
				}
			}

			//ppan+ worker informs the master of completion of a epoch
			MPI_Send(&ei, sizeof(size_t), MPI_BYTE, mpi_root, tagWorkerEopch, MPI_COMM_WORLD);
		}

		MPI_Wait(&prRequest, MPI_STATUS_IGNORE);
		std::cout << "worker " << rankId << " exit" << std::endl;
    }
    else
    {
        MPI_Request prRequests[nPartialResultsToUpdateWeights];
        MPI_Status prStatuses[nPartialResultsToUpdateWeights];

        MPI_Request wbRequests[nWorkers];
        MPI_Status wbStatuses[nWorkers];

        ByteBuffer partialResultMasterBuffer(0);
        {
            /* Receive the length of the buffer for serialized partial results */
            MPI_Request request;
            MPI_Irecv(&partialResultArchLength, sizeof(size_t), MPI_BYTE, MPI_ANY_SOURCE,
                       partialResultLengthTag, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            partialResultMasterBuffer.resize(nPartialResultsToUpdateWeights * partialResultArchLength);
        }

		for (size_t ei = 0; ei < nEpoch; ei++)
		{
			for (size_t i = 0; i < nSamples; i += batchSizeLocal)
			{
				/* Receive partial results from worker nodes */
				for (size_t i = 0; i < nPartialResultsToUpdateWeights; i++)
				{
					MPI_Irecv(&partialResultMasterBuffer[i * partialResultArchLength], partialResultArchLength, MPI_BYTE, MPI_ANY_SOURCE, partialResultTag,
						MPI_COMM_WORLD, &prRequests[i]);
				}

				for (size_t i = 0; i < nPartialResultsToUpdateWeights; i++)
				{
					int nodeIndex;
					/* Receive partial result from worker node */
					MPI_Waitany(nPartialResultsToUpdateWeights, prRequests, &nodeIndex, MPI_STATUS_IGNORE);
					prRequests[nodeIndex] = MPI_REQUEST_NULL;

					training::PartialResultPtr partialResult = deserializePartialResultFromNode(partialResultArchLength, &partialResultMasterBuffer[nodeIndex * partialResultArchLength]);

					/* Pass computed weights and biases derivatives to the master algorithm */
					netMaster.input.add(training::partialResults, i, partialResult);
				}

				/* Perform the step of optimization algorithm using partial results from worker nodes
				   to update weights and biases on master node */
				netMaster.compute();

				if (ei > 0 || i > 0)	//not the first batch
				{
					MPI_Waitall(nWorkers, wbRequests, wbStatuses);
				}

				if (ei + 1 < nEpoch || i + batchSizeLocal < nSamples)	//not the last batch
				{
					/* Send the updated weights and biases to worker nodes */
					training::ModelPtr wbModel = netMaster.getPartialResult()->get(training::resultFromMaster)->get(training::model);
					checkPtr((void *)wbModel.get());
					NumericTablePtr wb = wbModel->getWeightsAndBiases();
					/* Serialize weights and biases */
					InputDataArchive wbDataArch;
					wb->serialize(wbDataArch);
					wbDataArch.copyArchiveToArray(&wbBuffer[0], wbArchLength);

#if 0
{
	training::ModelPtr wbModel = netMaster.getPartialResult()->get(training::resultFromMaster)->get(training::model);
	checkPtr((void *)wbModel.get());
	NumericTablePtr wb = wbModel->getWeightsAndBiases(0);
	static NumericTablePtr owb;
	if (i + 100*batchSizeLocal >= nSamples)
	if (owb && wb)
	{
		covariance::Batch<> algorithm;
		algorithm.input.set(covariance::data, wb);
		algorithm.compute();
		services::SharedPtr<covariance::Result> res = algorithm.getResult();
		printf("xxx %d, %d\n", ei, i);
		printNumericTable(wb, "getWeightsAndBiases:");
		#if 1
		printNumericTable(res->get(covariance::covariance), "Covariance matrix:");
		printNumericTable(res->get(covariance::mean),       "Mean vector:");
		#endif
	}

	if (wb) owb = wb;
}
#endif
					for (size_t i = 0; i < nWorkers; i++)
					{
						MPI_Isend(&wbBuffer[0], wbArchLength, MPI_BYTE, i + 1, wbTag, MPI_COMM_WORLD, &wbRequests[i]);
					}
				}
        	}
		}
        /* Finalize neural network training on the master node */
        netMaster.finalizeCompute();

        /* Retrieve training and prediction models of the neural network */
        training::ModelPtr trModel = netMaster.getResult()->get(training::model);
        checkPtr((void*)trModel.get());
        predictionModel = trModel->getPredictionModel<float>();
    }
}

void testModel()
{
    /* Read testing data set from a .csv file and create a tensor to store input data */
    TensorPtr predictionData = readTensorFromCSV(testDatasetFile);

    /* Create an algorithm to compute the neural network predictions */
    prediction::Batch<> net;

    /* Set the batch size for the neural network prediction */
    net.parameter.batchSize = predictionData->getDimensionSize(0);

    /* Set input objects for the prediction neural network */
    net.input.set(prediction::model, predictionModel);
    net.input.set(prediction::data, predictionData);

    /* Run the neural network prediction */
    net.compute();

    /* Print results of the neural network prediction */
    predictionResult = net.getResult();
}

void printResults()
{
    /* Read testing ground truth from a .csv file and create a tensor to store the data */
    TensorPtr predictionGroundTruth = readTensorFromCSV(testGroundTruthFile);

    printTensors<int, float>(predictionGroundTruth, predictionResult->get(prediction::prediction),
                             "Ground truth", "Neural network predictions: each class probability",
                             "Neural network classification results (first 20 observations):", 20);
	//ppan+ calculate prediction precision over all test samples (ref. service.h)
	TensorPtr dataTable1 = predictionGroundTruth;
	TensorPtr dataTable2 = predictionResult->get(prediction::prediction);
	const daal::services::Collection<size_t> &dims = dataTable1->getDimensions();
	size_t nRows = dims[0];

	SubtensorDescriptor<float> block1;
	SubtensorDescriptor<float> block2;
	dataTable1->getSubtensor(0, 0, 0, nRows, readOnly, block1);
	dataTable2->getSubtensor(0, 0, 0, nRows, readOnly, block2);

	size_t nCols1 = block1.getSize() / nRows;
	size_t nCols2 = block2.getSize() / nRows;
	float *dataType1 = block1.getPtr();
	float *dataType2 = block2.getPtr();

	#define DO_REGRESSION	//undefine to do classification

	#ifdef DO_REGRESSION
	float err = 0;
	for (size_t i = 0; i < nRows; i++)
	{
		float pval = 0;
		for (size_t j = 0; j < nCols2; j++)
			pval += j * dataType2[i * nCols2 + j];

		err += fabs(pval - dataType1[i * nCols1 + 0]);
	}

	std::cout << "trained samples: " << ntrain << std::endl;
	std::cout << "tested  samples: " << nRows << std::endl;
	std::cout << "mean square err: " << std::setprecision(5) << err / 100 / nRows << std::endl;
	#else	//classification
	int nok = 0;
	for (size_t i = 0; i < nRows; i++)
	{
		float maxp = 0;
		size_t maxi = 0;

		for (size_t j = 0; j < nCols2; j++)
			if (dataType2[i * nCols2 + j] >= maxp)
			{
				maxi = j;
				maxp = dataType2[i * nCols2 + j];
			}

		if (maxi == dataType1[i * nCols1 + 0])
			++nok;
	}

	std::cout << "trained samples: " << ntrain << std::endl;
	std::cout << "tested  samples: " << nRows  << std::endl;
	std::cout << "matched samples: " << nok    << std::endl;
	std::cout << "precision: " 		 << 100 * nok / nRows << std::endl;
	#endif

    dataTable1->releaseSubtensor(block1);
    dataTable2->releaseSubtensor(block2);
}

void sendPartialResultToMaster(const training::PartialResultPtr &partialResult,
    size_t &partialResultArchLength,
    ByteBuffer &partialResultBuffer,
    MPI_Request &prRequest)
{
    InputDataArchive dataArch;
    NumericTablePtr wbDer = partialResult->get(training::derivatives);
    partialResult->serialize( dataArch );

    if (partialResultArchLength == 0)
    {
        partialResultArchLength = dataArch.getSizeOfArchive();
        partialResultBuffer.resize(partialResultArchLength);
    }

    dataArch.copyArchiveToArray(&partialResultBuffer[0], partialResultArchLength);

    MPI_Isend(&partialResultBuffer[0], partialResultArchLength, MPI_BYTE, mpi_root, partialResultTag, MPI_COMM_WORLD, &prRequest);
}

training::PartialResultPtr deserializePartialResultFromNode(
    size_t &partialResultArchLength, byte *partialResultBuffer)
{
    /* Deserialize partial results from step 1 */
    OutputDataArchive dataArch(partialResultBuffer, partialResultArchLength);

    training::PartialResultPtr partialResult(new training::PartialResult());
    partialResult->deserialize(dataArch);
    return partialResult;
}

NumericTablePtr pullWeightsAndBiasesFromMaster(size_t &wbArchLength, ByteBuffer &wbBuffer, MPI_Request &wbRequest)
{
    MPI_Wait(&wbRequest, MPI_STATUS_IGNORE);

    /* Deserialize weights and biases */
    OutputDataArchive wbDataArchLocal(&wbBuffer[0], wbArchLength);

    NumericTablePtr wbLocal(new HomogenNumericTable<float>());

    wbLocal->deserialize(wbDataArchLocal);

    return wbLocal;
}
