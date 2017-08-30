# this script split my 'universal' training data set ../data/etl2M.csv to 4 pieces
# then it builds and runs neural_net_dense_allgather_distributed_mpi2 to train 
# a Deep Neural Network for network performance prediction. 

DDIR=../data
INF=$DDIR/etl2M.csv
NL=`wc -l $INF | cut -d' ' -f1`
NN=`expr \( $NL - 10000 \) / 4`

## round train files to 100X lines
NN=`expr $NN / 100 \* 100`

[ $INF -nt $DDIR/trainx_xaa.csv ] &&
{
	shuf $INF|split -l $NN

	for f in xaa xab xac xad
	do
		cut -d, -f1-17  $f > $DDIR/trainx_$f.csv
		cut -d, -f18    $f > $DDIR/trainy_$f.csv
	done

	cut -d, -f1-17 xae > $DDIR/testx.csv
	cut -d, -f18   xae > $DDIR/testy.csv
	
	rm xa?
}

# somehow make fails to rebuild on dirty neural_net_dense_allgather_distributed_mpi2.h
[ neural_net_dense_allgather_distributed_mpi2.h -nt neural_net_dense_allgather_distributed_mpi2.cpp ] &&
	touch neural_net_dense_allgather_distributed_mpi2.cpp 

pushd /data/intel/daal/examples/cpp/mpi
make libintel64 sample=neural_net_dense_allgather_distributed_mpi2 compiler=gnu threading=parallel mode=build &&
{
    popd
    nohup bash -c 'time mpirun -n 4 /data/intel/daal/examples/cpp/mpi/_results/gnu_intel64_a/neural_net_dense_allgather_distributed_mpi2.exe' > nohup.out.adnn
    tail -f nohup.out.adnn
}

