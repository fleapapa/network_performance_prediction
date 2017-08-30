/* file: neural_net_dense_allgather_distributed_mpi.cpp */
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
!******************************************************************************/
/* ppan+
      Copyright 2017 Peter Pan (pertaining to my Heuristic Interleaved Parameter Alternator algorithm)
      Copyright 2017 Peter Pan (pertaining to all my added code, including the "early stopper", see EARLY_STOP_BATCHS)
 */
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <limits>
#include <thread>
#include <memory>
#include <map>

#include <mpi.h>
#include "daal.h"
#include "service.h"
#include "neural_net_dense_allgather_distributed_mpi2.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::optimization_solver;
using namespace daal::services;

typedef std::vector<byte> ByteBuffer;
typedef std::vector<MPI_Request> RequestBuffer;
typedef optimization_solver::precomputed::Batch<float> ObjFunction;

const size_t nNodes   = 4;
const float invNNodes = 1.0f / nNodes;

/* Input data set parameters */
//ppan+ share the same data dir as other Network ML projects
#define DDIR "../data/"
const string trainDatasetFileNames[nNodes] =
{
    DDIR"trainx_xaa.csv",
    DDIR"trainx_xab.csv",
    DDIR"trainx_xac.csv",
    DDIR"trainx_xad.csv"
};
const string trainGroundTruthFileNames[nNodes] =
{
    DDIR"trainy_xaa.csv",
    DDIR"trainy_xab.csv",
    DDIR"trainy_xac.csv",
    DDIR"trainy_xad.csv"
};
string testDatasetFile     = DDIR"testx.csv";
string testGroundTruthFile = DDIR"testy.csv";

const size_t batchSizeLocal = 50;

const size_t nLayers = 4;

TensorPtr trainingData;
TensorPtr trainingGroundTruth;
training::ModelPtr trainingModel;
prediction::ResultPtr predictionResult;

int rankId, comm_size;
#define mpi_root 0

static size_t ntrain, ntest;	//num of train/test samples
static size_t epoch;

static std::unique_ptr<LayerIds> ids;

void initializeData();
void initializeNetwork(const int neurons[], const int nneurons);
void trainModel();
void testModel();
void printResults(const time_t tused, const int neurons[], const int nn);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

	//ppan+ load data only once
	initializeData();

	/*
		ran the following configurations of 3-hiddern-layer NN to learn
		how RMSE changes across the configs. then use grid (:not HIPA).
	 */
	#if 0
	int v_neurons[][3] = {
		{200, 100, 101},	//mean square err: 0.15121
		{400, 200, 101},	//mean square err: 0.15099
		{100, 200, 101},	//mean square err: 0.15061
		{100, 400, 101},	//mean square err: 0.14963
		{ 50, 400, 101},	//mean square err: 0.14943
		{ 20, 400, 101},	//mean square err: 0.14902
		{100, 600, 101},	//mean square err: 0.14878
		{ 50, 600, 101},	//mean square err: 0.14868
		{ 40, 600, 101},	//mean square err: 0.14868
		{ 50,1000, 101},	//mean square err: 0.14822
		{ 25,1000, 101},	//mean square err: 0.14817
		{ 20,2000, 101},	//mean square err: 0.14792
		{ 20,3000, 101},	//mean square err: 0.14792
		{0}
		};
	*/

	for (auto neurons: v_neurons)
	{
	#else
	for (int h1 =  20; h1 <=  100; h1 +=  10)
	for (int h2 = 200; h2 <= 3000; h2 += 200)
	{
		int neurons[3] = {h1, h2, 101};		//101 because performance ratio is 0% - 100%
	#endif
		if (neurons[0])
		{
			auto tstart = time(0);
			initializeNetwork(neurons, 3);
			trainModel();

			if(rankId == mpi_root)
			{
				testModel();
				printResults(time(0) - tstart, neurons, 3);
			}

			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

    MPI_Finalize();

    return 0;
}

//ppan+ load data only once
void initializeData()
{
	trainingData = readTensorFromCSV(trainDatasetFileNames[rankId]);
	trainingGroundTruth = readTensorFromCSV(trainGroundTruthFileNames[rankId], true);

	auto testGroundTruth = readTensorFromCSV(testGroundTruthFile, true);
	ntrain = nNodes * trainingData->getDimensions()[0];
	ntest  = testGroundTruth->getDimensions()[0];

	printf("trained samples: %lu\n", ntrain);
	printf("tested  samples: %lu\n", ntest);
}

void initializeNetwork(const int neurons[], const int nneurons)
{
    const Collection<size_t> &dataDims = trainingData->getDimensions();

    /* Configure the neural network model on worker nodes */
    training::Parameter parameter;
    parameter.batchSize = batchSizeLocal;

	//must new for each iteration
	ids = std::unique_ptr<LayerIds>(new LayerIds);

    training::TopologyPtr topology = configureNet(std::vector<int>(neurons, neurons + nneurons), ids.get());
    trainingModel.reset(new training::Model());
    trainingModel->initialize<float>(dataDims, *topology, &parameter);
}

void initializeOptimizationSolver(optimization_solver::sgd::Batch<float> &solver,
    SharedPtr<ObjFunction> &objFunction, const TensorPtr &tensor)
{
    if (tensor && tensor->getSize())
    {
        NumericTablePtr inputArgTable = createTableFromTensor(*tensor);

        solver.input.set(iterative_solver::inputArgument,  inputArgTable);

        SharedPtr<iterative_solver::Result> solverResult = solver.getResult();
        solverResult->set(iterative_solver::minimum, inputArgTable);

        NumericTablePtr derivTable(new HomogenNumericTable<float>(1, tensor->getSize(), NumericTable::doAllocate));

        objFunction.reset(new ObjFunction());
        SharedPtr<objective_function::Result> result = objFunction->getResult();
        result->set(objective_function::gradientIdx, derivTable);
        objFunction->setResult(result);

        /* Set learning rate for the optimization solver used in the neural network */
//ppan+ original learning rate 0.001 leads to BIG (blew-up) gradients and results in ALL NaN's !!!
//        (*(HomogenNumericTable<double>::cast(solver.parameter.learningRateSequence)))[0][0] = 0.001;
        (*(HomogenNumericTable<double>::cast(solver.parameter.learningRateSequence)))[0][0] = 0.0005;

        solver.parameter.function = objFunction;
        solver.parameter.nIterations = 1;
    }
}

void setNextBatchToModel(training::Model &model, const TensorPtr &dataBatch, const TensorPtr &groundTruthBatch)
{
    layers::forward::LayerIface *firstFwdLayer = model.getForwardLayer(0).get();
    layers::forward::Input *firstFwdLayerInput = firstFwdLayer->getLayerInput();
    firstFwdLayerInput->set(layers::forward::data, dataBatch);
    firstFwdLayer->getLayerResult()->setResultForBackward(firstFwdLayerInput);

    layers::forward::LayerIface *lastFwdLayer = model.getForwardLayer(nLayers-1).get();
    loss::forward::Input *lossInput = static_cast<loss::forward::Input *>(lastFwdLayer->getLayerInput());
    lossInput->set(layers::loss::forward::groundTruth, groundTruthBatch);
    lastFwdLayer->getLayerResult()->setResultForBackward(lossInput);
}

void updateParameters(optimization_solver::sgd::Batch<float> &solver,
            TensorPtr &tensor, ByteBuffer &buffer, size_t archLength, MPI_Request &request)
{
    if (archLength > 0 && request != MPI_REQUEST_NULL)
    {
        /* Wait for partial derivatives from all nodes */
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        /* Compute the sum of derivatives received from all nodes */
        SharedPtr<HomogenNumericTable<float> > sumTable = HomogenNumericTable<float>::cast(
                solver.parameter.function->getResult()->get(objective_function::gradientIdx));
        float *sumData = sumTable->getArray();
        size_t sumSize = sumTable->getNumberOfRows();
        for (size_t i = 0; i < sumSize; i++)
        {
            sumData[i] = 0.0f;
        }
        for (size_t node = 0; node < nNodes; node++)
        {
            /* Retrieve a partial derivative from a node */
            TensorPtr tensor = Tensor::cast(deserializeDAALObject(&buffer[node * archLength], archLength));
            SubtensorDescriptor<float> subtensor;
            tensor->getSubtensor(0, 0, 0, tensor->getDimensionSize(0), readOnly, subtensor);
            const float *data = subtensor.getPtr();
            for (size_t i = 0; i < sumSize; i++)
            {
                sumData[i] += data[i];
            }
            tensor->releaseSubtensor(subtensor);
        }
        /* Compute the average of derivatives received from all nodes */
        for (size_t i = 0; i < sumSize; i++)
        {
            sumData[i] *= invNNodes;
        }

        /* Update weights on all nodes by performing a step of optimization algorithm */
        solver.compute();

        NumericTable *minimumTable = solver.getResult()->get(iterative_solver::minimum).get();
        copyTableToTensor(*minimumTable, *tensor);
    }
}

void allgatherDerivatives(Tensor *tensor, ByteBuffer &buffer, ByteBuffer &bufferLocal, MPI_Request &request)
{
    if (tensor && tensor->getSize() > 0)
    {
        serializeDAALObject(tensor, bufferLocal);
        size_t bufSize = bufferLocal.size();
        if (buffer.size() == 0) { buffer.resize(bufferLocal.size() * nNodes); }

        /* Initiate asynchronous transfer of partial derivatives to all nodes */
        MPI_Iallgather(&bufferLocal[0], bufSize, MPI_BYTE, &buffer[0], bufSize, MPI_BYTE, MPI_COMM_WORLD, &request);
    }
}

void trainModel()
{
    TensorPtr wTensor[nLayers];                                 // tensors of weights of each layer
    TensorPtr bTensor[nLayers];                                 // tensors of biases of each layer
    SharedPtr<ObjFunction> wObjFunc[nLayers];                   // objective functions associated with weight derivatives
    SharedPtr<ObjFunction> bObjFunc[nLayers];                   // objective functions associated with bias derivatives
    optimization_solver::sgd::Batch<float> wSolver[nLayers];    // optimization solvers associated with weights of each layer
    optimization_solver::sgd::Batch<float> bSolver[nLayers];    // optimization solvers associated with biases of each layer

    /* Set input arguments for the optimization solvers */
    for (size_t l = 0; l < nLayers; l++)
    {
        layers::forward::Input *fwdInput = trainingModel->getForwardLayer(l)->getLayerInput();
        wTensor[l] = fwdInput->get(layers::forward::weights);
        bTensor[l] = fwdInput->get(layers::forward::biases);
        initializeOptimizationSolver(wSolver[l], wObjFunc[l], wTensor[l]);
        initializeOptimizationSolver(bSolver[l], bObjFunc[l], bTensor[l]);
    }

    ByteBuffer wDerBuffersLocal[nLayers];   // buffer for serialized weight derivatives on local node
    ByteBuffer bDerBuffersLocal[nLayers];   // buffer for serialized bias derivatives on local node
    ByteBuffer wDerBuffers[nLayers];        // buffer for serialized weight derivatives from all nodes
    ByteBuffer bDerBuffers[nLayers];        // buffer for serialized bias derivatives from all nodes

    std::vector<MPI_Request> wDerRequests(nLayers, MPI_REQUEST_NULL);
    std::vector<MPI_Request> bDerRequests(nLayers, MPI_REQUEST_NULL);

	size_t nSamples = trainingData->getDimensionSize(0);

	//ppan+ somehow it crash if nSamples is not exact times of batchSizeLocal
	nSamples = nSamples / batchSizeLocal * batchSizeLocal;

	#define EARLY_STOP_BATCHS (nSamples / batchSizeLocal / 2)
	#ifdef	EARLY_STOP_BATCHS
	int batchs = 0, epoch = 0;
	float gloss = std::numeric_limits<float>::max();	//min loss across nodes
	float lloss = std::numeric_limits<float>::max();	//min loss on local node

    for (int64_t i = 0; i < nSamples; i += batchSizeLocal)
	#else
    for (size_t i = 0; i < nSamples; i += batchSizeLocal)
    #endif
    {
        /* Pass a training data set and dependent values to the algorithm */
        setNextBatchToModel(*trainingModel, getNextSubtensor(trainingData, i, batchSizeLocal),
                                            getNextSubtensor(trainingGroundTruth, i, batchSizeLocal));
        /* FORWARD PASS */
        for (size_t l = 0; l < nLayers; l++)
        {
            /* Wait for updated derivatives from all nodes and update weights on local node
               using the optimization solver algorithm */
            updateParameters(wSolver[l], wTensor[l], wDerBuffers[l], wDerBuffersLocal[l].size(), wDerRequests[l]);
            updateParameters(bSolver[l], bTensor[l], bDerBuffers[l], bDerBuffersLocal[l].size(), bDerRequests[l]);

            /* Compute forward layer results */
            trainingModel->getForwardLayer(l)->compute();

			#if 0	//debug derivatives explosion
			if (0 == rankId && l + 1 < nLayers)
			{
				char title1[200], title2[200];
				sprintf(title1, "wTensor[%d] i=%d", l,i);
				sprintf(title2, "bTensor[%d] i=%d", l,i);

				if (wTensor[l] && bTensor[l])
					if (wTensor[l]->getDimensions().size() && bTensor[l]->getDimensions().size())
						printTensors<float, float>(wTensor[l], bTensor[l], title1, title2, "weights/biases:", 20);
					else
					if (!wTensor[l]->getDimensions().size())
						printf("xxx wTensor[%d]->getDimensions() i=%d\n", l, i);
					else
					if (!bTensor[l]->getDimensions().size())
						printf("xxx bTensor[%d]->getDimensions() i=%d\n", l, i);
			}
			#endif

        }

        /* BACKWARD PASS */
        for (int l = nLayers - 1; l >= 0; l--)
        {
            /* Compute weight and bias derivatives for the batch of inputs on worker nodes */
            SharedPtr<layers::backward::LayerIface> layer = trainingModel->getBackwardLayer(l);
            layer->compute();

            /* Start derivatives gathering on all nodes */
            layers::backward::ResultPtr layerResult = layer->getLayerResult();
            TensorPtr wDerTensor = layerResult->get(layers::backward::weightDerivatives);
            TensorPtr bDerTensor = layerResult->get(layers::backward::biasDerivatives);

            allgatherDerivatives(wDerTensor.get(), wDerBuffers[l], wDerBuffersLocal[l], wDerRequests[l]);
            allgatherDerivatives(bDerTensor.get(), bDerBuffers[l], bDerBuffersLocal[l], bDerRequests[l]);

			#if 0	//debug derivatives explosion
			if (0 == rankId && l + 1 < nLayers && 0 == i%10000)
			{
				char title1[200], title2[200];
				sprintf(title1, "wDerTensor[%d] i=%d", l,i);
				sprintf(title2, "bDerTensor[%d] i=%d", l,i);

				if (wDerTensor && bDerTensor)
					if (wDerTensor->getDimensions().size() && bDerTensor->getDimensions().size())
						printTensors<float, float>(wDerTensor, bDerTensor, title1, title2, "weights/biases:", 20);
					else
					if (!wDerTensor->getDimensions().size())
						printf("xxx wDerTensor[%d]->getDimensions() i=%d\n", l, i);
					else
					if (!bDerTensor->getDimensions().size())
						printf("xxx bDerTensor[%d]->getDimensions() i=%d\n", l, i);
			}
			#endif
        }

		#if 1
		//loop till loss value does not improve for a certain number of batchs
		++batchs;
		if (i + batchSizeLocal >= nSamples)
		{
			++epoch;
			i = -batchSizeLocal;
		}

		//after at least one epoch, all nodes collect loss to decide early stop
		if (epoch > 0)
		{
			/* Get the result of the loss layer */
			services::SharedPtr<layers::forward::Result> lossLayerResult = trainingModel->getForwardLayer(ids.get()->sm)->getLayerResult();
			daal::services::SharedPtr<Tensor> ltensor = lossLayerResult->get(layers::forward::value);

			#if 0
			printf("rank %d: %d\n", rankId, i);
			printTensor(ltensor, "loss value: ");
			#endif

			const daal::services::Collection<size_t> &dims = ltensor->getDimensions();
			if (0 == dims.size()) continue;
			SubtensorDescriptor<float> block;
			ltensor->getSubtensor(0, 0, 0, 1, readOnly, block);
			lloss = std::min<float>(lloss, block.getPtr()[0]);
			ltensor->releaseSubtensor(block);

			//sync up loss across all nodes on every EARLY_STOP_BATCHS batchs
			if (0 == batchs % EARLY_STOP_BATCHS)
			{
				printf("lloss[%d, %d] = %.2f\n", rankId, i, lloss);

				float losses[nNodes];
				MPI_Allgather(&lloss, sizeof lloss, MPI_BYTE, losses, sizeof lloss, MPI_BYTE, MPI_COMM_WORLD);

				for (auto loss: losses)
					lloss = std::min<float>(lloss, loss);

				printf("gloss[%d] = %.2f\n", epoch, lloss);

				if (gloss > lloss)
					gloss = lloss;
				else
					break;	//early stop
			}
		}

		#endif
    }

    for (int l = nLayers - 1; l >= 0; l--)
    {
        /* Wait for updated derivatives from all nodes and update weights and biases on local node
           using the optimization solver algorithm */
        updateParameters(wSolver[l], wTensor[l], wDerBuffers[l], wDerBuffersLocal[l].size(), wDerRequests[l]);
        updateParameters(bSolver[l], bTensor[l], bDerBuffers[l], bDerBuffersLocal[l].size(), bDerRequests[l]);
    }
}

void testModel()
{
    /* Retrieve prediction model of the neural network */
    prediction::ModelPtr predictionModel = trainingModel->getPredictionModel<float>();

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

void printResults(const time_t tused, const int neurons[], const int nn)
{
	//ppan+ write truths and predictions to a file for neural_net_dense_allgather_distributed_mpi2.ipynb
	//to draw charts for the NN that has top 10 least RMSE.
	char ofdir[] = DDIR"/tmp";
	mkdir(ofdir, 0775);

	char ofname[200];
	sprintf(ofname, "%d", tused);
	for (int i = 0; i < nn; i++)
		sprintf(ofname + strlen(ofname), "_%d", neurons[i]);
	strcat(ofname, ".csv");

	char ofpath[200];
	sprintf(ofpath, "%s/%s", ofdir, ofname);
	FILE* of = fopen(ofpath, "w");

    /* Read testing ground truth from a .csv file and create a tensor to store the data */
    TensorPtr predictionGroundTruth = readTensorFromCSV(testGroundTruthFile);

	#if 1
    printTensors<int, float>(predictionGroundTruth, predictionResult->get(prediction::prediction),
                             "Ground truth", "Neural network predictions: each class probability",
                             "Neural network classification results (first 20 observations):", 20);
	#endif

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

	float errs = 0;
	for (size_t i = 0; i < nRows; i++)
	{
		float truth = dataType1[i * nCols1 + 0];
		float predictionv = 0;

		#if 1
		/*
		 * Funny is when prediction is based on weighted distribution of all probability,
		 * predicted values mostly crowd in range [60, 70]. To avoid such a undesired
		 * "smooth" predition, let's try only weighted distribution of top NTOP probabilities.
		 */
		#define NTOP 3
		struct PV{
			int v;		//prediction in [0, 100] (=network performance ratio)
			float p;	//probability
			PV(const int v, const float p):v(v), p(p){}
		};
		std::vector<PV> pv;

		for (size_t j = 0; j < nCols2; j++)
			pv.push_back(PV(j, dataType2[i * nCols2 + j]));

		//sort for top NTOP
		std::sort(pv.begin(), pv.end(), [](const PV& a, const PV& b) -> bool
		{
			return a.p > b.p;
		});

		//normalize probabilities of the top's
		float psum = 0;
		for (int i = 0; i < NTOP; i++)
			psum += pv[i].p;

		std::map<int, float> pvmap;
		for (int i = 0; i < NTOP; i++)
			pvmap[pv[i].v] = pv[i].p / psum;

		//compute prediction according to top's
		for (const auto &pv: pvmap)
			predictionv += pv.first * pv.second;
		#else
		for (size_t j = 0; j < nCols2; j++)
			predictionv += j * dataType2[i * nCols2 + j];
		#endif

		float err = (predictionv - truth);
		errs += err * err;

		if (of) fprintf(of, "%f,%f\n", truth, predictionv);
	}

	float rmse = sqrt(errs / nRows) / 100;
	printf("mean square errs: %.5f\n", rmse);
	printf("       time used: %d secs\n", tused);

    dataTable1->releaseSubtensor(block1);
    dataTable2->releaseSubtensor(block2);

	if (of) fclose(of);
	//rename for neural_net_dense_allgather_distributed_mpi2.ipynb to sort for top 10
	char nfpath[200];
	sprintf(nfpath, "%s/%.5f_%s", ofdir, rmse, ofname);
	rename(ofpath, nfpath);
}
