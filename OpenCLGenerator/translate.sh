if [ $# != 5 ]; then
  echo usage: ./translate.sh class strided device text-output binary-output
  exit 1
fi

# example: ./translate.sh PairwiseSimilarity64\$PairwiseCombiner f c fields.dump fields.binary

java -Djava.library.path=${APARAPI_HOME}/com.amd.aparapi.jni/dist -Dcom.amd.aparapi.enableShowGeneratedOpenCL=true -Dcom.amd.aparapi.associateWithHJ=true OpenCLGenerator $1 $2 $3
cp fields.dump $4
cp kernel.bin $5
