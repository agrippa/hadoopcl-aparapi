#include "JNIContext.h"
#include "OpenCLJNI.h"
#include "List.h"

JNIContext::JNIContext(JNIEnv *jenv, jobject _kernelObject,
        jint setTaskId,
        jint setAttemptId, jint setContextId): 
      kernelObject(jenv->NewGlobalRef(_kernelObject)),
      kernelClass((jclass)jenv->NewGlobalRef(jenv->GetObjectClass(_kernelObject))), 
      profileBaseTime(0),
      passes(0),
      profileFile(NULL) {

   srand (time(NULL));
   taskId = setTaskId;
   attemptId = setAttemptId;
   contextId = setContextId;
   datactx = NULL;
   kernelLaunchCounter = 0;
   dump_filename = (char *)malloc(512);
   currentLabel = NULL;
}

void JNIContext::dispose(JNIEnv *jenv) {
   cl_int status = CL_SUCCESS;
   jenv->DeleteGlobalRef(kernelObject);
   jenv->DeleteGlobalRef(kernelClass);
   free(dump_filename);
   if (argc > 0){
      for (int i=0; i< argc; i++){
         KernelArg *arg = args[i];
         if (!arg->isPrimitive()){
            if (arg->arrayBuffer != NULL){
               if (arg->arrayBuffer->mem != 0){
                  status = clReleaseMemObject((cl_mem)arg->arrayBuffer->mem);
                  CLException::checkCLError(status, "clReleaseMemObject()");
                  arg->arrayBuffer->mem = (cl_mem)0;
               }
               if (arg->arrayBuffer->javaArray != NULL)  {
                  jenv->DeleteWeakGlobalRef((jweak)arg->arrayBuffer->javaArray); // HERE
                  arg->arrayBuffer->javaArray = NULL;
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
   }
}
