#ifndef OPENCL_CONTEXT_H
#define OPENCL_CONTEXT_H

#include "Common.h"
#include "Config.h"

typedef struct _OpenCLContext {
    public:
        cl_device_id deviceId; // initOpenCL
        cl_int deviceType; // initOpenCL
        cl_context context; // initOpenCL
        cl_command_queue commandQueue; // buildProgramJNI
        cl_event prevExecEvent;
} OpenCLContext;

#endif
