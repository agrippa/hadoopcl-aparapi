#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>

#define CHECK(func_call) { cl_int err; if ( (err = (func_call)) != CL_SUCCESS) { fprintf(stderr, "Error %d at %s:%d\n", err, __FILE__, __LINE__); exit(1); } }
#define CHECK_ERR(my_err) { if (my_err != CL_SUCCESS) { fprintf(stderr, "Error %d at %s:%d\n", my_err, __FILE__, __LINE__); exit(1); } }

char *varsToPrint[] = { "isGPU", "nPairs", "nKeys", "nVals", "individualInputValsCount", "outputLength", "outputAuxIntLength", "outputAuxDoubleLength", "\0" };

typedef struct _Arg {
    char *name;
    char *type;
    size_t len;
    void *data;
    int isref;
} Arg;

static int isVariableToPrint(const char *name) {
    int i = 0;
    while (strlen(varsToPrint[i]) > 0) {
        if (strcmp(varsToPrint[i], name) == 0) {
            return i;
        }
        i++;
    }
    return -1;
}

static int sizeOfType(char *typeName) {
    if (strncmp(typeName, "int", 3) == 0) {
        return 4;
    } else if (strncmp(typeName, "long", 4) == 0) {
        return 8;
    } else if (strncmp(typeName, "float", 5) == 0) {
        return 4;
    } else if (strncmp(typeName, "double", 6) == 0) {
        return 8;
    } else {
        fprintf(stderr, "Unsupported variable type %s\n", typeName);
        exit(1);
    }
}

static void printVar(Arg *a) {
    if (!(a->isref)) {
        if (strncmp(a->type, "int", 3) == 0) {
            printf("%s has value %d\n", a->name, *((int *)(a->data)));
        } else if (strncmp(a->type, "long", 4) == 0) {
            printf("%s has value %l\n", a->name, *((long *)(a->data)));
        } else if (strncmp(a->type, "float", 5) == 0) {
            printf("%s has value %f\n", a->name, *((float *)(a->data)));
        } else if (strncmp(a->type, "double", 6) == 0) {
            printf("%s has value %f\n", a->name, *((double *)(a->data)));
        } else {
            fprintf(stderr, "Unsupported variable type %s\n", a->type);
            exit(1);
        }
    } else {
        fprintf(stderr, "Don't support printing vector types yet\n");
        exit(1);
    }
}

static void reliableRead(void *ptr, size_t size, size_t count, FILE *fp) {
    size_t soFar = 0;
    size_t oldSoFar = 0;
    while (soFar < count) {
        soFar += fread(((char *)ptr) + (soFar * size), size, count - soFar, fp);
        if (soFar == oldSoFar) {
            fprintf(stderr, "Read seems stuck at %llu\n", soFar);
        }
        oldSoFar = soFar;
    }
}

static int getDataTypeLength(char *type, char *name) {
    if (strncmp(type, "int", 3) == 0 || strncmp(type, "float", 5) == 0) {
        return 4;
    } else if (strncmp(type, "long", 4) == 0 || strncmp(type, "double", 6) == 0) {
        return 8;
    } else {
        fprintf(stderr, "Unsupported type %s for arg %s\n", type, name);
        // exit(1);
    }
}

static void printArg(Arg *a) {
    fprintf(stderr, "type=\"%s\" (%d bytes), name=\"%s\", isref=%d, length in bytes = %llu, data = %p\n", a->type, getDataTypeLength(a->type, a->name), a->name, a->isref, a->len, a->data);
}

static void readArg(Arg *out, FILE *in, int verbose) {
    int hasData;

    out->type = (char *)malloc(128);
    int soFar = 0;
    do {
        reliableRead(out->type + soFar, 1, 1, in);
    } while ((out->type)[soFar++] != '\0');

    out->name = (char *)malloc(128);
    soFar = 0;
    do {
        reliableRead(out->name + soFar, 1, 1, in);
    } while ((out->name)[soFar++] != '\0');

    reliableRead(&(out->len), sizeof(size_t), 1, in);

    reliableRead(&(out->isref), sizeof(int), 1, in);
    reliableRead(&hasData, sizeof(int), 1, in);

    if (hasData) {
        out->data = malloc(out->len);
    } else {
        out->data = NULL;
    }

    if (verbose) {
        printArg(out);
    }

    if (hasData) {
        reliableRead(out->data, getDataTypeLength(out->type, out->name), out->len / getDataTypeLength(out->type, out->name), in);
    }
}

static cl_uint getDevices(cl_platform_id platform, cl_device_type device_type, cl_device_id **devs) {
    cl_uint numDevices;
    CHECK(clGetDeviceIDs(platform, device_type, 0, NULL, &numDevices));
    *devs = (cl_device_id *)malloc(sizeof(cl_device_id) * numDevices);
    CHECK(clGetDeviceIDs(platform, device_type, numDevices, *devs, NULL));
    return numDevices;
}

static cl_uint getPlatforms(cl_platform_id **platforms) {
    cl_uint numPlatforms;
    CHECK(clGetPlatformIDs(0, NULL, &numPlatforms));
    *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    CHECK(clGetPlatformIDs(numPlatforms, *platforms, NULL));
    return numPlatforms;
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

static cl_command_queue getCommandQueue(cl_context ctx, cl_device_id dev) {
    cl_int err;
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    CHECK_ERR(err);
    return q;
}

static cl_program createAndBuildProgram(cl_context ctx, const char *source, cl_device_id dev) {
    cl_int err;

    size_t len = strlen(source);
    cl_program program = clCreateProgramWithSource(ctx, 1, &source, &len, &err);
    CHECK_ERR(err);

    err = clBuildProgram(program, 1, &dev, "-O0 -g", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        CHECK(clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
        char *log = (char *)malloc(log_size + 1);
        CHECK(clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
        log[log_size] = '\0';
        fprintf(stderr, "%s\n", log);
        exit(1);
    }
    return program;
}

static cl_kernel getKernel(cl_program program, const char *kernel_name) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    CHECK_ERR(err);
    return kernel;
}

static cl_mem *constructMemObjects(Arg *arguments, int nArgs, cl_context ctx,
        cl_command_queue cmd, cl_kernel kernel, int verbose) {
    int i;
    cl_int err;
    cl_mem *bufs = (cl_mem *)malloc(sizeof(cl_mem) * nArgs);
    memset(bufs, 0x00, sizeof(cl_mem) * nArgs);

    fprintf(stderr, "Constructing mem objects\n");
    for (i = 0; i < nArgs; i++) {
        fprintf(stderr, " For %s\n", arguments[i].name);
        if (arguments[i].isref) {
            bufs[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, arguments[i].len, NULL, &err);
            CHECK_ERR(err);
            if (arguments[i].data != NULL) {
                CHECK(clEnqueueWriteBuffer(cmd, bufs[i], CL_TRUE, 0, arguments[i].len, arguments[i].data, 0, NULL, NULL));
            }
            CHECK(clSetKernelArg(kernel, i, sizeof(cl_mem), bufs + i));
        } else {
            CHECK(clSetKernelArg(kernel, i, arguments[i].len, arguments[i].data));
        }
    }
    fprintf(stderr, "Done constructing mem objects\n");

    return bufs;
}

static void printValue(Arg *a, void *buf) {
    int i;
    int isNWrites = 0;
    int isLookAside = 0;

    if (strncmp(a->name, "nWrites", 7) == 0) isNWrites = 1;
    if (strncmp(a->name, "inputValLookAsideBuffer", 23) == 0) isLookAside = 1;
    int lengthToPrint = a->len / sizeOfType(a->type);
    if (!isNWrites && !isLookAside && lengthToPrint > 100) lengthToPrint = 100;

    int count = 0;

    fprintf(stderr, "%s %s:\n", a->type, a->name);
    fprintf(stderr, "  ");
    for (i = 0; i < lengthToPrint; i++) {
        if (strncmp(a->type, "int", 3) == 0) {
            if (isNWrites && ((int *)buf)[i] >= 0) count++;
            fprintf(stderr, "%d",((int *)buf)[i]);
            if (isLookAside && i < lengthToPrint-1) fprintf(stderr, "(%d)", ((int *)buf)[i+1] - ((int *)buf)[i]); 
            fprintf(stderr, " ");
        } else if (strncmp(a->type, "long", 4) == 0) {
            fprintf(stderr, "%ld ",((long *)buf)[i]);
        } else if (strncmp(a->type, "float", 5) == 0) {
            fprintf(stderr, "%f ",((float *)buf)[i]);
        } else if (strncmp(a->type, "double", 6) == 0) {
            fprintf(stderr, "%f ",((double *)buf)[i]);
        } else {
            fprintf(stderr, "Unsupported variable type %s\n", a->type);
            exit(1);
        }
    }

    if (isNWrites) {
      fprintf(stderr, "  Count = %d / %d\n", count, (a->len / sizeOfType(a->type)));
    }
    fprintf(stderr, "\n");
}

static void checkInputs(Arg *arguments, int nArgs) {
    int i;
    for (i = 0; i < nArgs; i++) {
        if (arguments[i].isref) {
            void *copy = NULL;
            if (strncmp(arguments[i].name, "input", 5) == 0 ||
                    strncmp(arguments[i].name, "nWrites", 7) == 0) {
                printValue(arguments + i, arguments[i].data);
            }
            if (copy) free(copy);
        }
    }
}

static void checkOutputs(Arg *arguments, int nArgs, cl_mem *bufs, cl_command_queue cmd) {
    int i;

    for (i = 0; i < nArgs; i++) {
        if (arguments[i].isref) {
            void *copy = NULL;
            if (strncmp(arguments[i].name, "mem", 3) == 0 ||
                    strncmp(arguments[i].name, "output", 6) == 0 ||
                    strncmp(arguments[i].name, "nWrites", 7) == 0) {
                copy = malloc(arguments[i].len);
                fprintf(stderr, "reading %d for %s\n",arguments[i].len, arguments[i].name);
                CHECK(clEnqueueReadBuffer(cmd, bufs[i], CL_TRUE, 0, arguments[i].len,
                      copy, 0, NULL, NULL));
                printValue(arguments + i, copy);
            }
            if (copy) free(copy);
        }
    }
}

static void printKnownVariables(Arg *arguments, int nArgs) {
    char formatter[128];
    int i;
    for (i = 0 ; i < nArgs; i++) {
        Arg *curr = arguments + i;
        int knownIndex = isVariableToPrint(curr->name);
        if (knownIndex >= 0) {
            printVar(curr);
        }
    }
}

static void runOpenCL(Arg *arguments, int nArgs, char *source, cl_device_type device_type, int verbose) {
    cl_platform_id *platforms;
    cl_device_id *devs;
    cl_event exec_event;

    cl_uint numPlatforms = getPlatforms(&platforms);
    if (verbose) fprintf(stderr, "Got %d platforms\n", numPlatforms);
    cl_uint numDevices = getDevices(platforms[0], device_type, &devs);
    if (verbose) fprintf(stderr, "Got %d devices\n", numDevices);

    cl_context ctx = getContext(platforms[0], devs[0]);
    if (verbose) fprintf(stderr, "Done creating context\n");
    cl_command_queue cmd = getCommandQueue(ctx, devs[0]);
    if (verbose) fprintf(stderr, "Done creating command queue\n");
    cl_program program = createAndBuildProgram(ctx, source, devs[0]);
    if (verbose) fprintf(stderr, "Done creating program\n");
    cl_kernel kernel = getKernel(program, "run");
    if (verbose) fprintf(stderr, "Done creating kernel\n");
    cl_mem *bufs = constructMemObjects(arguments, nArgs, ctx, cmd, kernel, verbose);

    size_t global_size = 128;
    size_t local_size = 128;

    if (verbose) checkInputs(arguments, nArgs);

    fprintf(stderr, "Launching kernel\n");
    CHECK(clEnqueueNDRangeKernel(cmd, kernel, 1, NULL, &global_size,
          &local_size, 0, NULL, &exec_event));
    fprintf(stderr, "  Done launching kernel\n");

    CHECK(clWaitForEvents(1, &exec_event));

    if (verbose) {
        printKnownVariables(arguments, nArgs);
        checkOutputs(arguments, nArgs, bufs, cmd);
    }
    fprintf(stderr, "  Done waiting for events\n");
}

static cl_device_type parseDeviceType(const char *dev_str) {
    if (strncmp(dev_str, "gpu", 3) == 0) {
        return CL_DEVICE_TYPE_GPU;
    } else if (strncmp(dev_str, "cpu", 3) == 0) {
        return CL_DEVICE_TYPE_CPU;
    } else {
        fprintf(stderr, "Unsupported device type %s\n", dev_str);
        exit(1);
    }
}

void scanForText(char *filename) {
    char c;
    char *buf = (char *)malloc(512);
    int len = 512;
    FILE *in = fopen(filename, "r");
    int nskipped = 0;
    int ninarow = 0;
    while (fread(&c, 1, 1, in) == 1) {
        if ((c >= ' ' && c <= '}') || c == '\n') {
            if (nskipped > 0) {
                //printf("---%d---", nskipped);
            }
            if (len-2 == ninarow) {
                buf = (char *)realloc(buf, len * 2);
                len = len * 2;
            }
            buf[ninarow++] = c;
            nskipped = 0;
        } else {
          if (ninarow > 2) {
            buf[ninarow] = '\0';
            printf("%s", buf);
          }
          ninarow = 0;
          nskipped++;
        }
    }
    free(buf);
    fclose(in);
}

int main(int argc, char **argv) {
    int i;
    char *source = NULL;

    if (argc < 4) {
        printf("usage: %s dump-file device-type verbose <kernel-file>\n", argv[0]);
        return 0;
    }

    // scanForText(argv[1]);
    FILE *in = fopen(argv[1], "r");
    if (in == NULL) {
        fprintf(stderr, "Failed opening %s\n", argv[1]);
        return 1;
    }
    cl_device_type device_type = parseDeviceType(argv[2]);
    int verbose = atoi(argv[3]);

    if (argc > 4) {
        FILE *kernelfile = fopen(argv[4], "r");
        fseek(kernelfile, 0L, SEEK_END);
        size_t sourceLength = ftell(kernelfile);
        fseek(kernelfile, 0L, SEEK_SET);

        source = (char *)malloc(sourceLength + 1);
        reliableRead(source, 1, sourceLength, kernelfile);
        source[sourceLength] = '\0';
        fclose(kernelfile);
    }

    int nArgs;
    reliableRead(&nArgs, sizeof(int), 1, in);
    if (verbose) {
        printf("%d arguments\n", nArgs);
    }

    Arg *arguments = (Arg *)malloc(sizeof(Arg) * nArgs);
    for (i = 0; i < nArgs; i++) {
        readArg(arguments + i, in, verbose);
    }

    int sourceLength;
    reliableRead(&sourceLength, sizeof(int), 1, in);

    if (source == NULL) {
        source = (char *)malloc(sourceLength + 1);
        reliableRead(source, 1, sourceLength + 1, in);
    } else {
        fseek(in, sourceLength + 1, SEEK_CUR);
    }

    if (verbose) {
        printf("%s\n", source);
    }

    char endMarker[5];
    reliableRead(endMarker, 1, 4, in);
    endMarker[4] = '\0';
    if (strncmp(endMarker, "DONE", 4) != 0) {
        fprintf(stderr, "Unexpected value %s for endMarker\n", endMarker);
        exit(1);
    }

    fclose(in);

    runOpenCL(arguments, nArgs, source, device_type, verbose);
    printf("Done!\n");

    return 0;
}
