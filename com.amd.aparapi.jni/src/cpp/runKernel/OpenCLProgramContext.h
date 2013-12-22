#ifndef OPENCL_PROGRAM_CONTEXT_H
#define OPENCL_PROGRAM_CONTEXT_H

#include "Common.h"
#include "Config.h"

typedef struct _OpenCLProgramContext {
    public:
      cl_kernel kernel;
      cl_program program;
} OpenCLProgramContext;

#endif
