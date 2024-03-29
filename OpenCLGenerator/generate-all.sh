#!/bin/sh

export CLASSPATH=${HADOOP_APP_DIR}/PairwiseSimilarity64.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/MahoutKMeans.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/TestWritables.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/FuzzyKMeans.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/NaiveBayes.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/Dirichlet.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/PiHCL2.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/SortHCL2.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/BlackscholesHCL2.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/KMeansHCL2.jar:${CLASSPATH}

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

# ./translate.sh TestWritables\$TestWritableMapper f c ~/kernels/writable.mapper.cpu ./bin
# ./translate.sh TestWritables\$TestWritableMapper t g ~/kernels/writable.mapper.gpu ./bin
# 
# ./translate.sh TestWritables\$TestWritableReducer f c ~/kernels/writable.reducer.cpu ./bin
# ./translate.sh TestWritables\$TestWritableReducer t g ~/kernels/writable.reducer.gpu ./bin

./translate.sh NaiveBayes\$NaiveBayesMapper f c ~/kernels/bayes.mapper.cpu ./bin
./translate.sh NaiveBayes\$NaiveBayesMapper t g ~/kernels/bayes.mapper.gpu ./bin

./translate.sh NaiveBayes\$NaiveBayesCombiner f c ~/kernels/bayes.combiner.cpu ./bin
./translate.sh NaiveBayes\$NaiveBayesCombiner t g ~/kernels/bayes.combiner.gpu ./bin

./translate.sh NaiveBayes\$NaiveBayesReducer f c ~/kernels/bayes.reducer.cpu ./bin
./translate.sh NaiveBayes\$NaiveBayesReducer t g ~/kernels/bayes.reducer.gpu ./bin

./translate.sh FuzzyKMeans\$FuzzyKMeansMapper f c ~/kernels/fuzzy.mapper.cpu ./bin
./translate.sh FuzzyKMeans\$FuzzyKMeansMapper t g ~/kernels/fuzzy.mapper.gpu ./bin

./translate.sh FuzzyKMeans\$FuzzyKMeansCombiner f c ~/kernels/fuzzy.combiner.cpu ./bin
./translate.sh FuzzyKMeans\$FuzzyKMeansCombiner t g ~/kernels/fuzzy.combiner.gpu ./bin

./translate.sh FuzzyKMeans\$FuzzyKMeansReducer f c ~/kernels/fuzzy.reducer.cpu ./bin
./translate.sh FuzzyKMeans\$FuzzyKMeansReducer t g ~/kernels/fuzzy.reducer.gpu ./bin

./translate.sh Dirichlet\$DirichletMapper f c ~/kernels/dirichlet.mapper.cpu ./bin
./translate.sh Dirichlet\$DirichletMapper t g ~/kernels/dirichlet.mapper.gpu ./bin

# Old kernels
./translate.sh PiHCL2\$PiHCL2Mapper f c ~/kernels/pi.mapper.cpu ./bin
./translate.sh PiHCL2\$PiHCL2Mapper t g ~/kernels/pi.mapper.gpu ./bin

./translate.sh PiHCL2\$PiHCL2Reducer f c ~/kernels/pi.reducer.cpu ./bin
./translate.sh PiHCL2\$PiHCL2Reducer t g ~/kernels/pi.reducer.gpu ./bin

./translate.sh BlackscholesHCL2\$BlackscholesHCL2Mapper f c ~/kernels/blackscholes.mapper.cpu ./bin
./translate.sh BlackscholesHCL2\$BlackscholesHCL2Mapper t g ~/kernels/blackscholes.mapper.gpu ./bin

./translate.sh BlackscholesHCL2\$BlackscholesHCL2Reducer f c ~/kernels/blackscholes.reducer.cpu ./bin
./translate.sh BlackscholesHCL2\$BlackscholesHCL2Reducer t g ~/kernels/blackscholes.reducer.gpu ./bin

./translate.sh SortHCL2\$SortHCL2Mapper f c ~/kernels/sort.mapper.cpu ./bin
./translate.sh SortHCL2\$SortHCL2Mapper t g ~/kernels/sort.mapper.gpu ./bin

./translate.sh SortHCL2\$SortHCL2Reducer f c ~/kernels/sort.reducer.cpu ./bin
./translate.sh SortHCL2\$SortHCL2Reducer t g ~/kernels/sort.reducer.gpu ./bin

./translate.sh KMeansHCL2\$KMeansHCL2Mapper f c ~/kernels/old_kmeans.mapper.cpu ./bin
./translate.sh KMeansHCL2\$KMeansHCL2Mapper t g ~/kernels/old_kmeans.mapper.gpu ./bin

./translate.sh KMeansHCL2\$KMeansHCL2Reducer f c ~/kernels/old_kmeans.reducer.cpu ./bin
./translate.sh KMeansHCL2\$KMeansHCL2Reducer t g ~/kernels/old_kmeans.reducer.gpu ./bin
