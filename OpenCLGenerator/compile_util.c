#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/timeb.h>

#define CHECK(func_call) { cl_int err; if ( (err = (func_call)) != CL_SUCCESS) { fprintf(stderr, "Error %d at %s:%d\n", err, __FILE__, __LINE__); exit(1); } }
#define CHECK_ERR(my_err) { if (my_err != CL_SUCCESS) { fprintf(stderr, "Error %d at %s:%d\n", my_err, __FILE__, __LINE__); exit(1); } }

unsigned long milliseconds(){
    struct timeb tm; 
    ftime(&tm);
    return (1000 * tm.time) + tm.millitm;
}

static cl_uint getPlatforms(cl_platform_id **platforms) {
    cl_uint numPlatforms;
    CHECK(clGetPlatformIDs(0, NULL, &numPlatforms));
    *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    CHECK(clGetPlatformIDs(numPlatforms, *platforms, NULL));
    return numPlatforms;
}

static cl_uint getDevices(cl_platform_id* platforms, int nplatforms,
        cl_platform_id *platform, cl_device_type device_type,
        cl_device_id **devs) {
    int i;
    cl_uint numDevices;
    for (i = 0; i < nplatforms; i++) {
        if (clGetDeviceIDs(platforms[i], device_type, 0, NULL, &numDevices) ==
                CL_DEVICE_NOT_FOUND) {
            continue;
        }
        *platform = platforms[i];
        break;
    }
    *devs = (cl_device_id *)malloc(sizeof(cl_device_id) * numDevices);
    CHECK(clGetDeviceIDs(*platform, device_type, numDevices, *devs, NULL));
    return numDevices;
}

static cl_context getContext(cl_platform_id platform, cl_device_id device) {
    cl_int status;
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM,
      (cl_context_properties)platform, 0 };
    cl_context ctx = clCreateContext( cps, 1, &device, NULL, NULL,
        &status);
    CHECK_ERR(status);
    return ctx;
}

int main(int argc, char **argv) {
    int i, j;
    if (argc != 2) {
        fprintf(stderr, "usage: %s filename\n", argv[0]);
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    size_t filelen;
    fseek(fp, 0L, SEEK_END);
    filelen = ftell(fp);
    fseek(fp, 0L, SEEK_SET);

    char *source = (char *)malloc(filelen + 1);
    if (fread(source, 1, filelen, fp) != filelen) {
        fprintf(stderr, "Error reading from file\n");
        return 1;
    }
    source[filelen] = '\0';
    fclose(fp);

    int index = 0;
    while (1) {
        if (source[index + 1] == '\n' && source[index] == '\n') {
            source = source + (index + 1);
            filelen = filelen - (index + 1);
            break;
        }
        index++;
    }

    cl_platform_id *platforms;
    cl_uint num_platforms = getPlatforms(&platforms);

    cl_platform_id cpu_platform, gpu_platform;
    cl_device_id *cpus, *gpus;
    cl_device_id cpu, gpu;

    cl_uint num_cpus = getDevices(platforms, num_platforms, &cpu_platform,
            CL_DEVICE_TYPE_CPU, &cpus);
    if (num_cpus <= 0) {
        fprintf(stderr, "Could not find CPUs\n");
        return 1;
    }
    cpu = cpus[0];

    cl_uint num_gpus = getDevices(platforms, num_platforms, &gpu_platform,
            CL_DEVICE_TYPE_GPU, &gpus);
    if (num_gpus <= 0) {
        fprintf(stderr, "Could not find GPUs\n");
        return 1;
    }
    gpu = gpus[0];

    cl_context cpu_context = getContext(cpu_platform, cpu);
    cl_context gpu_context = getContext(gpu_platform, gpu);

    cl_int err;
    cl_program cpu_program = clCreateProgramWithSource(cpu_context, 1, &source,
            &filelen, &err);
    CHECK_ERR(err);

    cl_program gpu_program = clCreateProgramWithSource(gpu_context, 1, &source,
            &filelen, &err);
    CHECK_ERR(err);

    long start = milliseconds();
    CHECK(clBuildProgram(cpu_program, 1, &cpu, NULL, NULL, NULL));
    CHECK(clBuildProgram(gpu_program, 1, &gpu, NULL, NULL, NULL));
    long stop = milliseconds();

    int linecount = 0;
    index = 0;
    while (source[index] != '\0') {
        if (source[index] == '\n') {
            linecount++;
        }
        index++;
    }

    printf("%ld ms to compile, %d lines\n", (stop - start) / 2, linecount);

    return 0;
}
