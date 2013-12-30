#include "KernelArg.h"
#include "JNIContext.h"
#include <string>
#include <iostream>

using std::string;
using std::cerr;
using std::endl;

jclass KernelArg::argClazz=(jclass)0;
jfieldID KernelArg::nameFieldID=0;
jfieldID KernelArg::typeFieldID=0; 
jfieldID KernelArg::javaArrayFieldID=0; 
jfieldID KernelArg::sizeInBytesFieldID=0;
jfieldID KernelArg::numElementsFieldID=0; 

KernelArg::KernelArg(JNIEnv *jenv, JNIContext *jniContext, jobject argObj):
   jniContext(jniContext),
   argObj(argObj){
      javaArg = jenv->NewGlobalRef(argObj);   // save a global ref to the java Arg Object
      if (argClazz == 0){
         jclass c = jenv->GetObjectClass(argObj); 
         nameFieldID = JNIHelper::GetFieldID(jenv, c, "name", "Ljava/lang/String;");
         typeFieldID = JNIHelper::GetFieldID(jenv, c, "type", "I");
         javaArrayFieldID = JNIHelper::GetFieldID(jenv, c, "javaArray", "Ljava/lang/Object;");
         sizeInBytesFieldID = JNIHelper::GetFieldID(jenv, c, "sizeInBytes", "J");
         numElementsFieldID = JNIHelper::GetFieldID(jenv, c, "numElements", "I");
         argClazz  = c;
      }
      type = jenv->GetIntField(argObj, typeFieldID);
      jstring nameString  = (jstring)jenv->GetObjectField(argObj, nameFieldID);
      const char *nameChars = jenv->GetStringUTFChars(nameString, NULL);
      if (strncmp(nameChars, "input", 5) == 0 || strncmp(nameChars, "globals", 7) == 0) {
          //fprintf(stderr,"Creating argument %s with direction IN\n",nameChars);
          dir = IN;
      } else if (strncmp(nameChars, "output", 6) == 0) {
          //fprintf(stderr,"Creating argument %s with direction OUT\n",nameChars);
          dir = OUT;
      } else {
          //fprintf(stderr,"Creating argument %s with direction INOUT\n",nameChars);
          dir = INOUT;
      }

      cachedValue = NULL;
      cachedValueLength = 0;

      if (strncmp(nameChars, "mem", 3) == 0) {
          zeroBeforeKernel = 1;
      } else {
          zeroBeforeKernel = 0;
      }

      name = strdup(nameChars);
      // fprintf(stderr,"Initializing %s\n",name);
      jenv->ReleaseStringUTFChars(nameString, nameChars);
      if (isArray()){
         arrayBuffer = new ArrayBuffer();
      } else if(isAparapiBuffer()) {
         aparapiBuffer = AparapiBuffer::flatten(jenv, argObj, type);
      }
   }

cl_int KernelArg::setLocalBufferArg(JNIEnv *jenv, int argIdx, int argPos, bool verbose) {
   if (verbose){
       fprintf(stderr, "ISLOCAL, clSetKernelArg(jniContext->kernel, %d, %d, NULL);\n", argIdx, (int) arrayBuffer->lengthInBytes);
   }
   return(clSetKernelArg(jniContext->clprgctx.kernel, argPos, (int)arrayBuffer->lengthInBytes, NULL));
}

cl_int KernelArg::setLocalAparapiBufferArg(JNIEnv *jenv, int argIdx, int argPos, bool verbose) {
   if (verbose){
       fprintf(stderr, "ISLOCAL, clSetKernelArg(jniContext->kernel, %d, %d, NULL);\n", argIdx, (int) aparapiBuffer->lengthInBytes);
   }
   return(clSetKernelArg(jniContext->clprgctx.kernel, argPos, (int)aparapiBuffer->lengthInBytes, NULL));
}

const char* KernelArg::getTypeName() {
   string s = "";
   if(isStatic()) {
      s += "static ";
   }
   if (isFloat()) {
      s += "float";
   }
   else if(isInt()) {
      s += "int";
   }
   else if(isBoolean()) {
      s += "boolean";
   }
   else if(isByte()) {
      s += "byte";
   }
   else if(isLong()) {
      s += "long";
   }
   else if(isDouble()) {
      s += "double";
   }
   return s.c_str();
}

void KernelArg::getPrimitiveValue(JNIEnv *jenv, jfloat* value) {
   jfieldID fieldID = jenv->GetFieldID(jniContext->kernelClass, name, "F");
   *value = jenv->GetFloatField(jniContext->kernelObject, fieldID);
}
void KernelArg::getPrimitiveValue(JNIEnv *jenv, jint* value) {
   jfieldID fieldID = jenv->GetFieldID(jniContext->kernelClass, name, "I");
   *value = jenv->GetIntField(jniContext->kernelObject, fieldID);
}
void KernelArg::getPrimitiveValue(JNIEnv *jenv, jboolean* value) {
   jfieldID fieldID = jenv->GetFieldID(jniContext->kernelClass, name, "B");
   *value = jenv->GetByteField(jniContext->kernelObject, fieldID);
}
void KernelArg::getPrimitiveValue(JNIEnv *jenv, jbyte* value) {
   jfieldID fieldID = jenv->GetFieldID(jniContext->kernelClass, name, "B");
   *value = jenv->GetByteField(jniContext->kernelObject, fieldID);
}
void KernelArg::getPrimitiveValue(JNIEnv *jenv, jlong* value) {
   jfieldID fieldID = jenv->GetFieldID(jniContext->kernelClass, name, "J");
   *value = jenv->GetLongField(jniContext->kernelObject, fieldID);
}
void KernelArg::getPrimitiveValue(JNIEnv *jenv, jdouble* value) {
   jfieldID fieldID = jenv->GetFieldID(jniContext->kernelClass, name, "D");
   *value = jenv->GetDoubleField(jniContext->kernelObject, fieldID);
}

void KernelArg::getStaticPrimitiveValue(JNIEnv *jenv, jfloat* value) {
   jfieldID fieldID = jenv->GetStaticFieldID(jniContext->kernelClass, name, "F");
   *value = jenv->GetStaticFloatField(jniContext->kernelClass, fieldID);
}
void KernelArg::getStaticPrimitiveValue(JNIEnv *jenv, jint* value) {
   jfieldID fieldID = jenv->GetStaticFieldID(jniContext->kernelClass, name, "I");
   *value = jenv->GetStaticIntField(jniContext->kernelClass, fieldID);
}
void KernelArg::getStaticPrimitiveValue(JNIEnv *jenv, jboolean* value) {
   jfieldID fieldID = jenv->GetStaticFieldID(jniContext->kernelClass, name, "Z");
   *value = jenv->GetStaticBooleanField(jniContext->kernelClass, fieldID);
}
void KernelArg::getStaticPrimitiveValue(JNIEnv *jenv, jbyte* value) {
   jfieldID fieldID = jenv->GetStaticFieldID(jniContext->kernelClass, name, "B");
   *value = jenv->GetStaticByteField(jniContext->kernelClass, fieldID);
}
void KernelArg::getStaticPrimitiveValue(JNIEnv *jenv, jlong* value) {
   jfieldID fieldID = jenv->GetStaticFieldID(jniContext->kernelClass, name, "J");
   *value = jenv->GetStaticLongField(jniContext->kernelClass, fieldID);
}
void KernelArg::getStaticPrimitiveValue(JNIEnv *jenv, jdouble* value) {
   jfieldID fieldID = jenv->GetStaticFieldID(jniContext->kernelClass, name, "D");
   *value = jenv->GetStaticDoubleField(jniContext->kernelClass, fieldID);
}

cl_int KernelArg::setPrimitiveArg(JNIEnv *jenv, int argIdx, int argPos, bool verbose, int useCached){
   cl_int status = CL_SUCCESS;

   if (useCached) {
       fprintf(stderr,"Setting %s to %d (len=%d)\n", name, *((int *)cachedValue), cachedValueLength);
       status = clSetKernelArg(jniContext->clprgctx.kernel, argPos, cachedValueLength, cachedValue);
   } else {
       if (isFloat()) {
           jfloat f;
           getPrimitive(jenv, argIdx, argPos, verbose, &f);

           cachedValue = (void *)realloc(cachedValue, sizeof(f));
           cachedValueLength = sizeof(f);
           memcpy(cachedValue, &f, sizeof(f));

           status = clSetKernelArg(jniContext->clprgctx.kernel, argPos, sizeof(f), &f);
       }
       else if (isInt()) {
           jint i;
           getPrimitive(jenv, argIdx, argPos, verbose, &i);

           cachedValue = (void *)realloc(cachedValue, sizeof(i));
           cachedValueLength = sizeof(i);
           memcpy(cachedValue, &i, sizeof(i));

           status = clSetKernelArg(jniContext->clprgctx.kernel, argPos, sizeof(i), &i);
       }
       else if (isBoolean()) {
           jboolean z;
           getPrimitive(jenv, argIdx, argPos, verbose, &z);

           cachedValue = (void *)realloc(cachedValue, sizeof(z));
           cachedValueLength = sizeof(z);
           memcpy(cachedValue, &z, sizeof(z));

           status = clSetKernelArg(jniContext->clprgctx.kernel, argPos, sizeof(z), &z);
       }
       else if (isByte()) {
           jbyte b;
           getPrimitive(jenv, argIdx, argPos, verbose, &b);

           cachedValue = (void *)realloc(cachedValue, sizeof(b));
           cachedValueLength = sizeof(b);
           memcpy(cachedValue, &b, sizeof(b));

           status = clSetKernelArg(jniContext->clprgctx.kernel, argPos, sizeof(b), &b);
       }
       else if (isLong()) {
           jlong l;
           getPrimitive(jenv, argIdx, argPos, verbose, &l);

           cachedValue = (void *)realloc(cachedValue, sizeof(l));
           cachedValueLength = sizeof(l);
           memcpy(cachedValue, &l, sizeof(l));

           status = clSetKernelArg(jniContext->clprgctx.kernel, argPos, sizeof(l), &l);
       }
       else if (isDouble()) {
           jdouble d;
           getPrimitive(jenv, argIdx, argPos, verbose, &d);

           cachedValue = (void *)realloc(cachedValue, sizeof(d));
           cachedValueLength = sizeof(d);
           memcpy(cachedValue, &d, sizeof(d));

           status = clSetKernelArg(jniContext->clprgctx.kernel, argPos, sizeof(d), &d);
       }
   }
   return status;
}


