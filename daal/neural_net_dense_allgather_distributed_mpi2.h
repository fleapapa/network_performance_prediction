/* file: neural_net_dense_distributed_mpi.h */
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

/* ppan+
      Copyright 2017 Peter Pan (pertaining to my Heuristic Interleaved Parameter Alternator algorithm)
      Copyright 2017 Peter Pan (pertaining to all my added code, including the early stopper)
 */
#include <vector>

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::services;

struct LayerIds
{
	std::vector<size_t> fc;
	size_t sm;
};

training::TopologyPtr configureNet(const std::vector<int> &neurons, LayerIds* ids = 0)
{
	// Create topology of the neural network
	training::TopologyPtr topology(new training::Topology());

	// Create layers of the neural network
	for (auto n: neurons)
	{
		// Create fully-connected layer and initialize layer parameters
		SharedPtr<fullyconnected::Batch<>> fullyConnectedLayer(new fullyconnected::Batch<>(n));

		// In future we'll make initial weights and biass be HIPAble too :)
		fullyConnectedLayer->parameter.weightsInitializer.reset(new initializers::uniform::Batch<>(-0.001, 0.001));
		fullyConnectedLayer->parameter.biasesInitializer .reset(new initializers::uniform::Batch<>(-0.000, 1.000));

		// Add this layer to the topology of the neural network
		ids->fc.push_back(topology->add(fullyConnectedLayer));
		printf("ids->fc[%d]: id %d, neurons %d\n", ids->fc.size() -1 , ids->fc[ids->fc.size() -1 ], n);
	}

	// Create softmax layer and initialize layer parameters
	SharedPtr<loss::softmax_cross::Batch<> > softmaxCrossEntropyLayer(new loss::softmax_cross::Batch<>());
	ids->sm = topology->add(softmaxCrossEntropyLayer);

	// assembly the layers...why still need this after topology->add()'s ??!
	for (int i = 0; i < ids->fc.size(); i ++)
		if (i + 1 == ids->fc.size())
				topology->get(ids->fc[i]).addNext(ids->sm);
		else	topology->get(ids->fc[i]).addNext(ids->fc[i + 1]);

	return topology;
}

//ppan+ copy from old service.h, because mpi's service.h has no this function
void printTensor(daal::services::SharedPtr<Tensor> dataTable, const char *message = "",
                 size_t nPrintedRows = 0, size_t nPrintedCols = 0, size_t interval = 10)
{
    const daal::services::Collection<size_t> &dims = dataTable->getDimensions();
    size_t nRows = dims[0];

    if(nPrintedRows != 0)
    {
        nPrintedRows = std::min(nRows, nPrintedRows);
    }
    else
    {
        nPrintedRows = nRows;
    }

    SubtensorDescriptor<float> block;

    dataTable->getSubtensor(0, 0, 0, nPrintedRows, readOnly, block);

    size_t nCols = block.getSize() / nPrintedRows;

    if(nPrintedCols != 0)
    {
        nPrintedCols = std::min(nCols, nPrintedCols);
    }
    else
    {
        nPrintedCols = nCols;
    }

    printArray<float>(block.getPtr(), nPrintedCols, nPrintedRows, nCols, message, interval);

    dataTable->releaseSubtensor(block);
}
