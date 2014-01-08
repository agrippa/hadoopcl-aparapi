#include "OpenCLDataContext.h"
#include "JNIContext.h"

OpenCLDataContext::OpenCLDataContext() {
    hadoopclParams = NULL;
    nHadoopclParams = 0;
}

void OpenCLDataContext::cleanup() {
   for (int i = 0; i < nHadoopclParams; i++) {
       hadoopclParameter *current = hadoopclParams + i;
       clReleaseMemObject(current->allocatedMem);
   }
   if (hadoopclParams) free(hadoopclParams);
}

void OpenCLDataContext::printOpenclMemChecks() {
    int i;
    unsigned int sum = 0;

    fprintf(stderr,"Have allocated:\n");
    for (i = 0; i < nHadoopclParams; i++) {
        fprintf(stderr,"  %s: %llu bytes\n", hadoopclParams[i].name,
                hadoopclParams[i].allocatedSize);
        sum += hadoopclParams[i].allocatedSize;
    }
    fprintf(stderr,"    Total = %u bytes\n",sum);
}

hadoopclParameter* OpenCLDataContext::addHadoopclParam(KernelArg *arg, JNIContext *jniContext) {
    hadoopclParams = (hadoopclParameter *)realloc(hadoopclParams,
            sizeof(hadoopclParameter) * (nHadoopclParams + 1));
    hadoopclParameter *current = hadoopclParams + nHadoopclParams;
    nHadoopclParams++;

    if (!arg->isArray()) {
        fprintf(stderr,"Error: adding an arg that is not an array\n");
        exit(2);
    }

    current->name = (char *)malloc(sizeof(char) * (strlen(arg->name)+1));
    memcpy(current->name, arg->name, sizeof(char) * (strlen(arg->name)+1));
    current->allocatedSize = (size_t)arg->arrayBuffer->lengthInBytes;
    if (strncmp(current->name, "globals", 7) == 0) {
        current->createFlags = CL_MEM_READ_ONLY;
    } else {
        current->createFlags = CL_MEM_READ_WRITE;
    }
    // TODO depending on whether name contains input/output we should be
    // able to set more accurate flags here
    cl_int err;
    current->allocatedMem = clCreateBuffer(jniContext->clctx.context,
        current->createFlags, current->allocatedSize, NULL, &err);

    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error allocating buffer of size %llu for %s: %d\n",
            current->allocatedSize, current->name, err);
        exit(3);
    }
    // fprintf(stderr, "Adding param %s with size %llu\n",
    //         current->name, current->allocatedSize);
    return current;
}

hadoopclParameter* OpenCLDataContext::findHadoopclParam(KernelArg *arg) {
    int i;
    if (hadoopclParams == NULL) return NULL;

    for (i = 0; i < nHadoopclParams; i++) {
        hadoopclParameter *current = hadoopclParams + i;
        if (strcmp(arg->name, current->name) == 0) {
            return current;
        }
    }
    return NULL;
}

void OpenCLDataContext::refreshHadoopclParam(KernelArg *arg,
        hadoopclParameter *hadoopclParam, JNIContext *jniContext) {
    if (arg->arrayBuffer->lengthInBytes <= hadoopclParam->allocatedSize) {
        return;
    }
    
    // fprintf(stderr, "Refreshing param %s from %llu to %llu\n",
    //         hadoopclParam->name, hadoopclParam->allocatedSize,
    //         arg->arrayBuffer->lengthInBytes);

    cl_int err = clReleaseMemObject(hadoopclParam->allocatedMem);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error releasing memory during refresh of object %s\n",
                arg->name);
        exit(4);
    }
    hadoopclParam->allocatedMem = clCreateBuffer(jniContext->clctx.context,
            hadoopclParam->createFlags, arg->arrayBuffer->lengthInBytes,
            NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error allocating new memory buffer of size %llu "
                "during refresh of object %s\n",
                arg->arrayBuffer->lengthInBytes, arg->name);
        exit(5);
    }
    hadoopclParam->allocatedSize = arg->arrayBuffer->lengthInBytes;
}

cl_mem OpenCLDataContext::hadoopclRefresh(KernelArg *arg, int relaunch, JNIContext *jniContext) {
    if (!arg->isArray()) return 0x0;

    hadoopclParameter *param = findHadoopclParam(arg);

    if (relaunch) {
        if (param == NULL) {
            fprintf(stderr, "Failed finding param %s\n", arg->name);
            exit(1);
        }
        return param->allocatedMem;
    } else {
        if (param == NULL) {
            param = addHadoopclParam(arg, jniContext);
        }
        refreshHadoopclParam(arg, param, jniContext);
        return param->allocatedMem;
    }
}


