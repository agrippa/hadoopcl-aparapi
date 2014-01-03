#include "JNIContext.h"
#include "OpenCLJNI.h"
#include "List.h"

hadoopclParameter* JNIContext::addHadoopclParam(KernelArg *arg) {
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
    current->allocatedMem = clCreateBuffer(clctx.context,
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

hadoopclParameter* JNIContext::findHadoopclParam(KernelArg *arg) {
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

void JNIContext::refreshHadoopclParam(KernelArg *arg,
        hadoopclParameter *hadoopclParam) {
    if (arg->arrayBuffer->lengthInBytes <= hadoopclParam->allocatedSize) {
        return;
    }
    
    fprintf(stderr, "Refreshing param %s from %llu to %llu\n",
            hadoopclParam->name, hadoopclParam->allocatedSize,
            arg->arrayBuffer->lengthInBytes);

    cl_int err = clReleaseMemObject(hadoopclParam->allocatedMem);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error releasing memory during refresh of object %s\n",
                arg->name);
        exit(4);
    }
    hadoopclParam->allocatedMem = clCreateBuffer(clctx.context,
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

cl_mem JNIContext::hadoopclRefresh(KernelArg *arg, int relaunch) {
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
            param = addHadoopclParam(arg);
        }
        refreshHadoopclParam(arg, param);
        return param->allocatedMem;
    }
}

void JNIContext::printOpenclMemChecks() {
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

JNIContext::JNIContext(JNIEnv *jenv, jobject _kernelObject,
        jobject _openCLDeviceObject, jint _flags, jint setContextId): 
      kernelObject(jenv->NewGlobalRef(_kernelObject)),
      kernelClass((jclass)jenv->NewGlobalRef(jenv->GetObjectClass(_kernelObject))), 
      openCLDeviceObject(jenv->NewGlobalRef(_openCLDeviceObject)),
      flags(_flags),
      profileBaseTime(0),
      passes(0),
      exec(NULL),
      profileFile(NULL), 
      valid(JNI_FALSE){

   contextId = setContextId;
   memset(&clctx, 0x00, sizeof(OpenCLContext));
   hadoopclParams = NULL;
   nHadoopclParams = 0;
   valid = JNI_TRUE;
}

void JNIContext::dispose(JNIEnv *jenv, Config* config) {
   //fprintf(stdout, "dispose()\n");
   cl_int status = CL_SUCCESS;
   jenv->DeleteGlobalRef(kernelObject);
   jenv->DeleteGlobalRef(kernelClass);
   if (clctx.context != 0){
      status = clReleaseContext(clctx.context);
      //fprintf(stdout, "dispose context %0lx\n", context);
      CLException::checkCLError(status, "clReleaseContext()");
      clctx.context = (cl_context)0;
   }
   if (clctx.commandQueue != 0){
      if (config->isTrackingOpenCLResources()){
         commandQueueList.remove((cl_command_queue)clctx.commandQueue, __LINE__,
                 __FILE__);
      }
      status = clReleaseCommandQueue((cl_command_queue)clctx.commandQueue);
      //fprintf(stdout, "dispose commandQueue %0lx\n", commandQueue);
      CLException::checkCLError(status, "clReleaseCommandQueue()");
      clctx.commandQueue = (cl_command_queue)0;
   }
   if (clprgctx.program != 0){
      status = clReleaseProgram((cl_program)clprgctx.program);
      //fprintf(stdout, "dispose program %0lx\n", program);
      CLException::checkCLError(status, "clReleaseProgram()");
      clprgctx.program = (cl_program)0;
   }
   if (clprgctx.kernel != 0){
      status = clReleaseKernel((cl_kernel)clprgctx.kernel);
      //fprintf(stdout, "dispose kernel %0lx\n", kernel);
      CLException::checkCLError(status, "clReleaseKernel()");
      clprgctx.kernel = (cl_kernel)0;
   }
   if (argc > 0){
      for (int i=0; i< argc; i++){
         KernelArg *arg = args[i];
         if (!arg->isPrimitive()){
            if (arg->arrayBuffer != NULL){
               if (arg->arrayBuffer->mem != 0){
                  if (config->isTrackingOpenCLResources()){
                     memList.remove((cl_mem)arg->arrayBuffer->mem, __LINE__,
                             __FILE__);
                  }
                  status = clReleaseMemObject((cl_mem)arg->arrayBuffer->mem);
                  CLException::checkCLError(status, "clReleaseMemObject()");
                  arg->arrayBuffer->mem = (cl_mem)0;
               }
               if (arg->arrayBuffer->javaArray != NULL)  {
                  jenv->DeleteWeakGlobalRef((jweak)arg->arrayBuffer->javaArray);
               }
               delete arg->arrayBuffer;
               arg->arrayBuffer = NULL;
            }
         }
         if (arg->name != NULL){
            free(arg->name); arg->name = NULL;
         }
         if (arg->javaArg != NULL ) {
            jenv->DeleteGlobalRef((jobject) arg->javaArg);
         }
         delete arg; arg=args[i]=NULL;
      }
      delete[] args; args=NULL;

      // do we need to call clReleaseEvent on any of these that are still retained....
      delete[] readEvents; readEvents = NULL;
      delete[] writeEvents; writeEvents = NULL;
      delete[] executeEvents; executeEvents = NULL;

      if (config->isProfilingEnabled()) {
         if (config->isProfilingCSVEnabled()) {
            if (profileFile != NULL && profileFile != stderr) {
               fclose(profileFile);
            }
         }
         delete[] readEventArgs; readEventArgs=0;
         delete[] writeEventArgs; writeEventArgs=0;
      } 
   }
   if (config->isTrackingOpenCLResources()){
      fprintf(stderr, "after dispose{ \n");
      commandQueueList.report(stderr);
      memList.report(stderr); 
      readEventList.report(stderr); 
      executeEventList.report(stderr); 
      writeEventList.report(stderr); 
      fprintf(stderr, "}\n");
   }

   for (int i = 0; i < nHadoopclParams; i++) {
       hadoopclParameter *current = hadoopclParams + i;
       clReleaseMemObject(current->allocatedMem);
   }
   if (hadoopclParams) free(hadoopclParams);
}

void JNIContext::unpinAll(JNIEnv* jenv) {
   for (int i=0; i< argc; i++){
      KernelArg *arg = args[i];
      if (arg->isBackedByArray()) {
         arg->unpin(jenv);
      }
   }
}

