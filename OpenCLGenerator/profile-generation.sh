#!/bin/sh 
export CLASSPATH=${HADOOP_APP_DIR}/PairwiseSimilarity64.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/MahoutKMeans.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/TestWritables.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/FuzzyKMeans.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/NaiveBayes.jar:${CLASSPATH}
export CLASSPATH=${HADOOP_APP_DIR}/Dirichlet.jar:${CLASSPATH}

function profile
{
    SUM_TIME=0
    for i in {1..5}; do
        { time $* &> /dev/null ; } &> timing
        TIME=$(cat timing | awk '{ print $2 }' | tail -n 3  | head -n 1)
        LEN=$(echo ${#TIME} - 3 | bc)
        TIME=${TIME:2:$LEN}
        SUM_TIME=$(echo $SUM_TIME + $TIME | bc -l)
    done
    echo $(echo $SUM_TIME / 5 | bc -l)
}

function profile_compilation
{
    SUM_TIME=0
    for i in {1..5}; do
        TIME=$(./a.out $1 | awk '{ print $1 }')
        SUM_TIME=$(echo $SUM_TIME + $TIME | bc -l)
    done
    echo $(echo $SUM_TIME / 5 | bc -l)
}

function run_compilation_analysis
{
    APP=$1
    TASK=$2

    COMPILE_TIME=$(profile_compilation ~/kernels/$APP.$TASK.gpu)
    echo $APP $TASK compilation $COMPILE_TIME
    echo $APP $TASK linecount $(./a.out ~/kernels/$APP.$TASK.gpu | awk '{ print $5 }')
}

# kmeans
TIME=$(profile ./translate.sh MahoutKMeans\$MahoutKMeansMapper f c ~/kernels/kmeans.mapper.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh MahoutKMeans\$MahoutKMeansMapper t g ~/kernels/kmeans.mapper.gpu ./bin) | bc -l)
echo kmeans mapper translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis kmeans mapper

TIME=$(profile ./translate.sh MahoutKMeans\$MahoutKMeansReducer f c ~/kernels/kmeans.reducer.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh MahoutKMeans\$MahoutKMeansReducer t g ~/kernels/kmeans.reducer.gpu ./bin) | bc -l)
echo kmeans reducer translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis kmeans reducer

# fuzzy
TIME=$(profile ./translate.sh FuzzyKMeans\$FuzzyKMeansMapper f c ~/kernels/fuzzy.mapper.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh FuzzyKMeans\$FuzzyKMeansMapper t g ~/kernels/fuzzy.mapper.gpu ./bin) | bc -l)
echo fuzzy mapper translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis fuzzy mapper

TIME=$(profile ./translate.sh FuzzyKMeans\$FuzzyKMeansCombiner f c ~/kernels/fuzzy.combiner.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh FuzzyKMeans\$FuzzyKMeansCombiner t g ~/kernels/fuzzy.combiner.gpu ./bin) | bc -l)
echo fuzzy combiner translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis fuzzy combiner

TIME=$(profile ./translate.sh FuzzyKMeans\$FuzzyKMeansReducer f c ~/kernels/fuzzy.reducer.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh FuzzyKMeans\$FuzzyKMeansReducer t g ~/kernels/fuzzy.reducer.gpu ./bin) | bc -l)
echo fuzzy reducer translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis fuzzy reducer

# dirichlet
TIME=$(profile ./translate.sh Dirichlet\$DirichletMapper f c ~/kernels/dirichlet.mapper.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh Dirichlet\$DirichletMapper t g ~/kernels/dirichlet.mapper.gpu ./bin) | bc -l)
echo dirichlet mapper translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis dirichlet mapper

# pairwise
TIME=$(profile ./translate.sh PairwiseSimilarity64\$PairwiseMapper f c ~/kernels/pairwise.mapper.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh PairwiseSimilarity64\$PairwiseMapper t g ~/kernels/pairwise.mapper.gpu ./bin) | bc -l)
echo pairwise mapper translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis pairwise mapper

TIME=$(profile ./translate.sh PairwiseSimilarity64\$PairwiseCombiner f c ~/kernels/pairwise.combiner.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh PairwiseSimilarity64\$PairwiseCombiner t g ~/kernels/pairwise.combiner.gpu ./bin) | bc -l)
echo pairwise combiner translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis pairwise combiner

TIME=$(profile ./translate.sh PairwiseSimilarity64\$PairwiseReducer f c ~/kernels/pairwise.reducer.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh PairwiseSimilarity64\$PairwiseReducer t g ~/kernels/pairwise.reducer.gpu ./bin) | bc -l)
echo pairwise reducer translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis pairwise reducer

# bayes
TIME=$(profile ./translate.sh NaiveBayes\$NaiveBayesMapper f c ~/kernels/bayes.mapper.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh NaiveBayes\$NaiveBayesMapper t g ~/kernels/bayes.mapper.gpu ./bin) | bc -l)
echo bayes mapper translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis bayes mapper

TIME=$(profile ./translate.sh NaiveBayes\$NaiveBayesCombiner f c ~/kernels/bayes.combiner.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh NaiveBayes\$NaiveBayesCombiner t g ~/kernels/bayes.combiner.gpu ./bin) | bc -l)
echo bayes combiner translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis bayes combiner

TIME=$(profile ./translate.sh NaiveBayes\$NaiveBayesReducer f c ~/kernels/bayes.reducer.cpu ./bin)
TIME=$(echo $TIME + $(profile ./translate.sh NaiveBayes\$NaiveBayesReducer t g ~/kernels/bayes.reducer.gpu ./bin) | bc -l)
echo bayes reducer translation $(echo $TIME / 2 | bc -l)
run_compilation_analysis bayes reducer
