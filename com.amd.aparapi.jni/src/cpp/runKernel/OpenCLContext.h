#ifndef OPENCL_CONTEXT_H
#define OPENCL_CONTEXT_H

#include "Common.h"
#include "Config.h"

class OpenCLContext {
    public:
        cl_device_id deviceId;
        cl_int deviceType;
        cl_context context;
        cl_command_queue commandQueue;
        cl_program program;
        cl_kernel kernel;
}

#endif
