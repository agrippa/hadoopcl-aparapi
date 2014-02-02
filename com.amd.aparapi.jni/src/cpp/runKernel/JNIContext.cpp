#include "JNIContext.h"
#include "OpenCLJNI.h"
#include "List.h"

JNIContext::JNIContext(JNIEnv *jenv, jobject _kernelObject,
        jobject _openCLDeviceObject, jint _flags, jint setTaskId,
        jint setAttemptId, jint setContextId): 
      kernelObject(jenv->NewGlobalRef(_kernelObject)),
      kernelClass((jclass)jenv->NewGlobalRef(jenv->GetObjectClass(_kernelObject))), 
      openCLDeviceObject(jenv->NewGlobalRef(_openCLDeviceObject)),
      flags(_flags),
      profileBaseTime(0),
      passes(0),
      exec(NULL),
      profileFile(NULL), 
      valid(JNI_FALSE){

   srand (time(NULL));
   taskId = setTaskId;
   attemptId = setAttemptId;
   contextId = setContextId;
   datactx = NULL;
   kernelLaunchCounter = 0;
   memset(&clctx, 0x00, sizeof(OpenCLContext));
   valid = JNI_TRUE;
   dump_filename = (char *)malloc(512);
}

void JNIContext::dispose(JNIEnv *jenv, Config* config) {
   cl_int status = CL_SUCCESS;
   jenv->DeleteGlobalRef(kernelObject);
   jenv->DeleteGlobalRef(kernelClass);
   free(dump_filename);
   // if (clctx.context != 0){
   //    status = clReleaseContext(clctx.context);
   //    //fprintf(stdout, "dispose context %0lx\n", context);
   //    CLException::checkCLError(status, "clReleaseContext()");
   //    clctx.context = (cl_context)0;
   // }
   // if (clctx.execCommandQueue != 0){
   //    if (config->isTrackingOpenCLResources()){
   //       commandQueueList.remove((cl_command_queue)clctx.execCommandQueue, __LINE__,
   //               __FILE__);
   //    }
   //    status = clReleaseCommandQueue((cl_command_queue)clctx.execCommandQueue);
   //    //fprintf(stdout, "dispose commandQueue %0lx\n", commandQueue);
   //    CLException::checkCLError(status, "clReleaseCommandQueue()");
   //    clctx.execCommandQueue = (cl_command_queue)0;
   // }
   // if (clprgctx.program != 0){
   //    status = clReleaseProgram((cl_program)clprgctx.program);
   //    //fprintf(stdout, "dispose program %0lx\n", program);
   //    CLException::checkCLError(status, "clReleaseProgram()");
   //    clprgctx.program = (cl_program)0;
   // }
   // if (clprgctx.kernel != 0){
   //    status = clReleaseKernel((cl_kernel)clprgctx.kernel);
   //    //fprintf(stdout, "dispose kernel %0lx\n", kernel);
   //    CLException::checkCLError(status, "clReleaseKernel()");
   //    clprgctx.kernel = (cl_kernel)0;
   // }
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
}

void JNIContext::unpinAll(JNIEnv* jenv) {
   for (int i=0; i< argc; i++){
      KernelArg *arg = args[i];
      if (arg->isBackedByArray()) {
         arg->unpin(jenv);
      }
   }
}

