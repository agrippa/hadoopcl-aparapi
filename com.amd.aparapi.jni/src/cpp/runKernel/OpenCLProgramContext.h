#ifndef OPENCL_PROGRAM_CONTEXT_H
#define OPENCL_PROGRAM_CONTEXT_H

#include "Common.h"
#include "Config.h"
#include <pthread.h>

typedef struct _OpenCLProgramContext {
    public:
      cl_kernel kernel;
      cl_program program;
      char *source;
      pthread_mutex_t lock;
} OpenCLProgramContext;

#endif
