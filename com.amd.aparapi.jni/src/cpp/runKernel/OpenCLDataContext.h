#ifndef OPENCL_DATA_CONTEXT_H
#define OPENCL_DATA_CONTEXT_H

#include <CL/cl.h>
#include "KernelArg.h"

typedef struct _hadoopclParameter {
    char *name;
    cl_mem allocatedMem;
    size_t allocatedSize;
    cl_mem_flags createFlags;
} hadoopclParameter;

class OpenCLDataContext {
  public:
    hadoopclParameter *hadoopclParams;
    int nHadoopclParams;
    int writtenAtleastOnce;

    OpenCLDataContext();
    void cleanup();

    void printOpenclMemChecks();
    cl_mem hadoopclRefresh(KernelArg *arg, int relaunch, JNIContext *jniContext);
    hadoopclParameter* addHadoopclParam(KernelArg *arg, JNIContext *jniContext);
    hadoopclParameter* findHadoopclParam(KernelArg *arg);
    void refreshHadoopclParam(KernelArg *arg, hadoopclParameter *hadoopclParam, JNIContext *jniContext);

};

#endif
