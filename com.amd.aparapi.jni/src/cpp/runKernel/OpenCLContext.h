#ifndef OPENCL_CONTEXT_H
#define OPENCL_CONTEXT_H

#include "Common.h"

typedef struct _OpenCLContext {
    public:
        cl_device_id deviceId; // initOpenCL
        cl_context context; // initOpenCL
        cl_command_queue copyCommandQueue; // buildProgramJNI
        cl_command_queue execCommandQueue;
} OpenCLContext;

#endif
