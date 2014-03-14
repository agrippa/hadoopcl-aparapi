if [ $# != 5 ]; then
  echo usage: ./translate.sh class strided device text-output binary-output
  exit 1
fi

java -Djava.library.path=${APARAPI_HOME}/com.amd.aparapi.jni/dist -cp ${HADOOP_HOME}/build/hadoop-core-1.0.4-SNAPSHOT.jar:${APARAPI_HOME}/com.amd.aparapi/dist/aparapi.jar:${HOME}/Akihiro-Hayashi-Research-Project/OpenCLGenerator:$PWD:${MAHOUT_HOME}/integration/target/dependency/mahout-math-0.8-SNAPSHOT.jar:${MAHOUT_HOME}/integration/target/dependency/guava-14.0.1.jar:${MAHOUT_HOME}/core/target/mahout-core-0.8-SNAPSHOT.jar -Dcom.amd.aparapi.enableShowGeneratedOpenCL=true -Dcom.amd.aparapi.associateWithHJ=true OpenCLGenerator $1 $2 $3
cp fields.dump $4
cp kernel.bin $5
