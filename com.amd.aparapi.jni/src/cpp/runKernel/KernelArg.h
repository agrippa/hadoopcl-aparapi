
#ifndef KERNEL_ARG_H
#define KERNEL_ARG_H

#include "OpenCLContext.h"
#include "OpenCLProgramContext.h"
#include "Common.h"
#include "JNIHelper.h"
#include "ArrayBuffer.h"
#include "com_amd_aparapi_internal_jni_KernelRunnerJNI.h"
#include <iostream>

#ifdef _WIN32
#define strdup _strdup
#endif

class JNIContext;

enum KernelArgDirection {
    IN,
    OUT,
    GLOBAL,
    WRITABLE,
    INOUT
};

union kernel_arg {
    cl_mem mem;
    jint ji;
    jfloat jf;
    jboolean jbool;
    jbyte jb;
    jlong jl;
    jdouble jd;

    int i;
};

class KernelArg{
   private:
      static jclass argClazz;
      static jfieldID nameFieldID;
      static jfieldID typeFieldID; 
      static jfieldID sizeInBytesFieldID;
      static jfieldID numElementsFieldID;
      void *cachedValue;
      size_t cachedValueLength;

      const char* getTypeName();

      //all of these use JNIContext so they can't be inlined

      //get the value of a primitive arguement
      void getPrimitiveValue(JNIEnv *jenv, jfloat *value);
      void getPrimitiveValue(JNIEnv *jenv, jint *value);
      void getPrimitiveValue(JNIEnv *jenv, jboolean *value);
      void getPrimitiveValue(JNIEnv *jenv, jbyte *value);
      void getPrimitiveValue(JNIEnv *jenv, jlong *value);
      void getPrimitiveValue(JNIEnv *jenv, jdouble *value);

      //get the value of a static primitive arguement
      void getStaticPrimitiveValue(JNIEnv *jenv, jfloat *value);
      void getStaticPrimitiveValue(JNIEnv *jenv, jint *value);
      void getStaticPrimitiveValue(JNIEnv *jenv, jboolean *value);
      void getStaticPrimitiveValue(JNIEnv *jenv, jbyte *value);
      void getStaticPrimitiveValue(JNIEnv *jenv, jlong *value);
      void getStaticPrimitiveValue(JNIEnv *jenv, jdouble *value);

      template<typename T> 
      void getPrimitive(JNIEnv *jenv, int argIdx, int argPos, T* value) {
         if(isStatic()) {
            getStaticPrimitiveValue(jenv, value);
         } else {
            getPrimitiveValue(jenv, value);
         }
      }

      void dumpTypeToFile(FILE *fp);
      size_t getLengthForType();
      void dumpLengthInBytesToFile(FILE *fp, int relaunch, JNIEnv *jenv);
      void dumpData(FILE *fp, int relaunch, JNIEnv *jenv, JNIContext *jniContext, OpenCLContext *openclContext);


   public:
      static jfieldID javaArrayFieldID; 
   public:
      JNIContext *jniContext;  
      OpenCLProgramContext *programContext;
      jobject argObj;    // the Java KernelRunner.KernelArg object that we are mirroring.
      jobject javaArg;   // global reference to the corresponding java KernelArg object we grabbed our own global reference so that the object won't be collected until we dispose!
      char *name;        // used for debugging printfs
      jint type;         // a bit mask determining the type of this arg
      enum KernelArgDirection dir;
      int zeroBeforeKernel;

      ArrayBuffer *arrayBuffer;

      // Uses JNIContext so cant inline here see below
      KernelArg(JNIEnv *jenv, JNIContext *jniContext,
          OpenCLProgramContext *programContext, jobject argObj);

      ~KernelArg(){
          if (cachedValue) {
              free(cachedValue);
          }
      }

      void unpinAbort(JNIEnv *jenv){
         arrayBuffer->unpinAbort(jenv);
      }
      void unpinCommit(JNIEnv *jenv){
         arrayBuffer->unpinCommit(jenv);
      }
      void unpin(JNIEnv *jenv){
         //if  (value.ref.isPinned == JNI_FALSE){		 
         //     fprintf(stdout, "why are we unpinning buffer %s! isPinned = JNI_TRUE\n", name);
         //}
         // fprintf(stderr,"Unpinning %s\n",name);
         if (isMutableByKernel()){
            // we only need to commit if the buffer has been written to
            // we use mode=0 in that case (rather than JNI_COMMIT) because that frees any copy buffer if it exists
            // in most cases this array will have been pinned so this will not be an issue
            unpinCommit(jenv);
         }else {
            // fast path for a read_only buffer
            unpinAbort(jenv);
         }
         // fprintf(stderr," Success!\n",name);
      }
      void pin(JNIEnv *jenv){
         // fprintf(stderr,"Pinning %s\n",name);
         arrayBuffer->pin(jenv);
      }

      int isArray(){
         return(type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_ARRAY);
      }
      int isReadByKernel(){
         return(type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_READ);
      }
      int isMutableByKernel(){
         return(type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_WRITE);
      }
      int isExplicit(){
         return(type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_EXPLICIT);
      }
      int usesArrayLength(){
         return(type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_ARRAYLENGTH);
      }
      int isExplicitWrite(){
         return(type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_EXPLICIT_WRITE);
      }
      int isImplicit(){
         return(!isExplicit());
      }
      int isPrimitive(){
         return(type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_PRIMITIVE);
      }
      int isGlobal(){
         return(type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_GLOBAL);
      }
      int isFloat(){
         return(type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_FLOAT);
      }
      int isLong(){
         return (type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_LONG);
      }
      int isInt(){
         return (type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_INT);
      }
      int isDouble(){
         return (type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_DOUBLE);
      }
      int isBoolean(){
         return (type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_BOOLEAN);
      }
      int isByte(){
         return (type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_BYTE);
      }
      int isShort(){
         return (type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_SHORT);
      }
      int isLocal(){
         return (type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_LOCAL);
      }
      int isStatic(){
         return (type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_STATIC);
      }
      int isConstant(){
         return (type&com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_CONSTANT);
      }
      int isBackedByArray(){
         return ( (isArray() && (isGlobal() || isConstant())));
      }
      int needToEnqueueRead(){
         return(((isArray() && isGlobal())) && (isImplicit()&&isMutableByKernel()));
      }
      int needToEnqueueWrite(){
         return ((isImplicit()&&isReadByKernel())||(isExplicit()&&isExplicitWrite()));
      }
      void syncType(JNIEnv* jenv){
         type = jenv->GetIntField(javaArg, typeFieldID);
      }
      void syncSizeInBytes(JNIEnv* jenv){
         arrayBuffer->lengthInBytes = jenv->GetLongField(javaArg, sizeInBytesFieldID);
      }
      void syncJavaArrayLength(JNIEnv* jenv){
         arrayBuffer->length = jenv->GetIntField(javaArg, numElementsFieldID);
      }
      void clearExplicitBufferBit(JNIEnv* jenv){
         type &= ~com_amd_aparapi_internal_jni_KernelRunnerJNI_ARG_EXPLICIT_WRITE;
         jenv->SetIntField(javaArg, typeFieldID,type );
      }

      // Uses JNIContext so can't inline here we below.  
      void syncValue(JNIEnv *jenv);

      // Uses JNIContext so can't inline here we below.  
      cl_int setLocalBufferArg(JNIEnv *jenv, int argIdx, int argPos, bool verbose);
      // Uses JNIContext so can't inline here we below.  
      cl_int setPrimitiveArg(JNIEnv *jenv, int argIdx, int argPos, int useCached,
              kernel_arg *arg, int *size);
      void dumpToFile(FILE *fp, int relaunch, JNIEnv *jenv, JNIContext *jniContext, OpenCLContext *openclContext);
};


#endif // KERNEL_ARG_H
