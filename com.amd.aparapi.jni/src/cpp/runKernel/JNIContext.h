#ifndef JNI_CONTEXT_H
#define JNI_CONTEXT_H

#include "OpenCLDataContext.h"
#include "OpenCLProgramContext.h"
#include "OpenCLContext.h"
#include "Common.h"
#include "KernelArg.h"
#include "ProfileInfo.h"
#include "com_amd_aparapi_internal_jni_KernelRunnerJNI.h"
#include "Config.h"

class JNIContext {
private: 
   jint flags;
   jboolean valid;

public:

   jobject kernelObject;
   jobject openCLDeviceObject;
   jclass kernelClass;
   jint argc;
   KernelArg** args;
   cl_event* executeEvents;
   cl_event* readEvents;
   cl_ulong profileBaseTime;
   jint* readEventArgs;
   cl_event* writeEvents;
   jint* writeEventArgs;
   jboolean firstRun;
   jint passes;
   ProfileInfo *exec;
   FILE* profileFile;

   /*
    * HadoopCL stuff
    */
   OpenCLContext clctx;
   OpenCLProgramContext clprgctx;
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

   JNIContext(JNIEnv *jenv, jobject _kernelObject, jobject _openCLDeviceObject, jint _flags, jint setTaskId, jint setAttemptId, jint setContextId);
   
   static JNIContext* getJNIContext(jlong jniContextHandle){
      return((JNIContext*)jniContextHandle);
   }

   jboolean isValid(){
      return(valid);
   }

   jboolean isUsingGPU(){
      //I'm pretty sure that this is equivalend to:
      //return flags & com_amd_aparapi_internal_jni_KernelRunnerJNI_JNI_FLAG_USE_GPU;
      return((flags&com_amd_aparapi_internal_jni_KernelRunnerJNI_JNI_FLAG_USE_GPU)==com_amd_aparapi_internal_jni_KernelRunnerJNI_JNI_FLAG_USE_GPU?JNI_TRUE:JNI_FALSE);
   }

   ~JNIContext(){
   }

   void dispose(JNIEnv *jenv, Config* config);

   /**
    * Release JNI critical pinned arrays before returning to java code
    */
   void unpinAll(JNIEnv* jenv);
};

#endif // JNI_CONTEXT_H
