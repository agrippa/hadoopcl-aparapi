#ifndef JNI_CONTEXT_H
#define JNI_CONTEXT_H


#include "OpenCLContext.h"
#include "Common.h"
#include "KernelArg.h"
#include "ProfileInfo.h"
#include "com_amd_aparapi_internal_jni_KernelRunnerJNI.h"
#include "Config.h"

typedef struct _hadoopclParameter {
    char *name;
    cl_mem allocatedMem;
    size_t allocatedSize;
    cl_mem_flags createFlags;
} hadoopclParameter;

class JNIContext {
private: 
   jint flags;
   jboolean valid;

public:
   jobject kernelObject;
   jobject openCLDeviceObject;
   jclass kernelClass;
   OpenCLContext clctx;
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
   hadoopclParameter *hadoopclParams;
   int nHadoopclParams;
   cl_event exec_event;
   unsigned long startWrite;
   unsigned long stopWrite;
   unsigned long startKernel;

   cl_mem hadoopclRefresh(KernelArg *arg);
   hadoopclParameter* addHadoopclParam(KernelArg *arg);
   hadoopclParameter* findHadoopclParam(KernelArg *arg);
   void refreshHadoopclParam(KernelArg *arg, hadoopclParameter *hadoopclParam);
   void printOpenclMemChecks();

   JNIContext(JNIEnv *jenv, jobject _kernelObject, jobject _openCLDeviceObject, jint _flags);
   
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
