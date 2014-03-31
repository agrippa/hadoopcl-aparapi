#ifndef JNI_CONTEXT_H
#define JNI_CONTEXT_H

#include "OpenCLDataContext.h"
#include "OpenCLProgramContext.h"
#include "OpenCLContext.h"
#include "Common.h"
#include "KernelArg.h"
#include "com_amd_aparapi_internal_jni_KernelRunnerJNI.h"

class JNIContext {

public:
   jobject kernelObject;
   jclass kernelClass;
   jint argc;
   KernelArg** args;
   cl_ulong profileBaseTime;
   jint passes;
   FILE* profileFile;

   /*
    * HadoopCL stuff
    */
   OpenCLDataContext *datactx;
   cl_event exec_event;
   unsigned long startWrite;
   unsigned long stopWrite;
   unsigned long startKernel;
   char *dump_filename;
   unsigned int taskId;
   unsigned int attemptId;
   unsigned int contextId;
   unsigned int kernelLaunchCounter;
   char *currentLabel;

   JNIContext(JNIEnv *jenv, jobject _kernelObject, jint setTaskId, jint setAttemptId, jint setContextId);
   
   static JNIContext* getJNIContext(jlong jniContextHandle){
      return((JNIContext*)jniContextHandle);
   }

   ~JNIContext(){
   }

   void dispose(JNIEnv *jenv);
};

#endif // JNI_CONTEXT_H
