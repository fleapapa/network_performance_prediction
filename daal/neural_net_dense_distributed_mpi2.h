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
    size_t fc1;
    size_t fc2;
    size_t fc3;
    size_t sm;
};

training::TopologyPtr configureNet(LayerIds* ids = NULL)
{
    /* Create layers of the neural network */
    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<fullyconnected::Batch<> > fullyConnectedLayer1(new fullyconnected::Batch<>(200));

    fullyConnectedLayer1->parameter.weightsInitializer.reset(new initializers::uniform::Batch<>(-0.001, 0.001));

    fullyConnectedLayer1->parameter.biasesInitializer.reset(new initializers::uniform::Batch<>(0, 1));

    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<fullyconnected::Batch<> > fullyConnectedLayer2(new fullyconnected::Batch<>(100));

    fullyConnectedLayer2->parameter.weightsInitializer.reset(new initializers::uniform::Batch<>(-0.001, 0.001));

    fullyConnectedLayer2->parameter.biasesInitializer.reset(new initializers::uniform::Batch<>(0, 1));

    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<fullyconnected::Batch<> > fullyConnectedLayer3(new fullyconnected::Batch<>(101));	//5

    fullyConnectedLayer3->parameter.weightsInitializer.reset(new initializers::uniform::Batch<>(-0.001, 0.001));

    fullyConnectedLayer3->parameter.biasesInitializer.reset(new initializers::uniform::Batch<>(0, 1));

    /* Create softmax layer and initialize layer parameters */
    SharedPtr<loss::softmax_cross::Batch<> > softmaxCrossEntropyLayer(new loss::softmax_cross::Batch<>());

    /* Create topology of the neural network */
    training::TopologyPtr topology(new training::Topology());

    /* Add layers to the topology of the neural network */
    const size_t fc1 = topology->add(fullyConnectedLayer1);
    const size_t fc2 = topology->add(fullyConnectedLayer2);
    const size_t fc3 = topology->add(fullyConnectedLayer3);
    const size_t sm = topology->add(softmaxCrossEntropyLayer);
    topology->get(fc1).addNext(fc2);
    topology->get(fc2).addNext(fc3);
    topology->get(fc3).addNext(sm);
    if(ids)
    {
        ids->fc1 = fc1;
        ids->fc2 = fc2;
        ids->fc3 = fc3;
        ids->sm = sm;
    }
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
