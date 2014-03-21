#!/bin/sh

export CLASSPATH=${HADOOP_APP_DIR}/PairwiseSimilarity64.jar:${HADOOP_APP_DIR}/MahoutKMeans.jar:${CLASSPATH}

./translate.sh PairwiseSimilarity64\$PairwiseMapper f c ~/kernels/pairwise.mapper.cpu ./bin
./translate.sh PairwiseSimilarity64\$PairwiseMapper t g ~/kernels/pairwise.mapper.gpu ./bin

./translate.sh PairwiseSimilarity64\$PairwiseCombiner f c ~/kernels/pairwise.combiner.cpu ./bin
./translate.sh PairwiseSimilarity64\$PairwiseCombiner t g ~/kernels/pairwise.combiner.gpu ./bin

./translate.sh PairwiseSimilarity64\$PairwiseReducer f c ~/kernels/pairwise.reducer.cpu ./bin
./translate.sh PairwiseSimilarity64\$PairwiseReducer t g ~/kernels/pairwise.reducer.gpu ./bin

./translate.sh MahoutKMeans\$MahoutKMeansMapper f c ~/kernels/kmeans.mapper.cpu ./bin
./translate.sh MahoutKMeans\$MahoutKMeansMapper t g ~/kernels/kmeans.mapper.gpu ./bin

./translate.sh MahoutKMeans\$MahoutKMeansReducer f c ~/kernels/kmeans.reducer.cpu ./bin
./translate.sh MahoutKMeans\$MahoutKMeansReducer t g ~/kernels/kmeans.reducer.gpu ./bin
