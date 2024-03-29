#include "KernelArg.h"
#include "JNIContext.h"
#include <string>
#include <iostream>

using std::string;
using std::cerr;
using std::endl;

extern void reliableWrite(const void *ptr, size_t size, size_t count, FILE *fp);

jclass KernelArg::argClazz=(jclass)0;
jfieldID KernelArg::nameFieldID=0;
jfieldID KernelArg::typeFieldID=0; 
jfieldID KernelArg::javaArrayFieldID=0; 
jfieldID KernelArg::sizeInBytesFieldID=0;
jfieldID KernelArg::numElementsFieldID=0; 

KernelArg::KernelArg(JNIEnv *jenv, JNIContext *jniContext,
    OpenCLProgramContext *programContext, jobject argObj):
   jniContext(jniContext),
   programContext(programContext),
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
      if (strncmp(nameChars, "global", 6) == 0) {
          dir = GLOBAL;
      } else if (strncmp(nameChars, "writable", 8) == 0) {
          dir = WRITABLE;
      } else if (strncmp(nameChars, "input", 5) == 0) {
          dir = IN;
      } else if (strncmp(nameChars, "output", 6) == 0) {
          dir = OUT;
      } else {
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
      jenv->ReleaseStringUTFChars(nameString, nameChars);
      if (isArray()){
         arrayBuffer = new ArrayBuffer();
      }
   }

cl_int KernelArg::setLocalBufferArg(JNIEnv *jenv, int argIdx, int argPos, bool verbose) {
   return(clSetKernelArg(programContext->kernel, argPos,
         (int)arrayBuffer->lengthInBytes, NULL));
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

cl_int KernelArg::setPrimitiveArg(JNIEnv *jenv, int argIdx, int argPos, int useCached,
        kernel_arg *arg, int *size){
   cl_int status = CL_SUCCESS;

   if (useCached) {
       memcpy(arg, cachedValue, cachedValueLength);
       // status = clSetKernelArg(programContext->kernel, argPos,
       //     cachedValueLength, cachedValue);
   } else {
       if (isFloat()) {
           jfloat f;
           getPrimitive(jenv, argIdx, argPos, &f);

           cachedValue = (void *)realloc(cachedValue, sizeof(f));
           cachedValueLength = sizeof(f);
           memcpy(cachedValue, &f, sizeof(f));

           arg->jf = f;
           *size = sizeof(f);
           // status = clSetKernelArg(programContext->kernel, argPos, sizeof(f), &f);
       }
       else if (isInt()) {
           jint i;
           getPrimitive(jenv, argIdx, argPos, &i);

           cachedValue = (void *)realloc(cachedValue, sizeof(i));
           cachedValueLength = sizeof(i);
           memcpy(cachedValue, &i, sizeof(i));

           arg->ji = i;
           *size = sizeof(i);
           // status = clSetKernelArg(programContext->kernel, argPos, sizeof(i), &i);
       }
       else if (isBoolean()) {
           jboolean z;
           getPrimitive(jenv, argIdx, argPos, &z);

           cachedValue = (void *)realloc(cachedValue, sizeof(z));
           cachedValueLength = sizeof(z);
           memcpy(cachedValue, &z, sizeof(z));

           arg->jbool = z;
           *size = sizeof(z);
           // status = clSetKernelArg(programContext->kernel, argPos, sizeof(z), &z);
       }
       else if (isByte()) {
           jbyte b;
           getPrimitive(jenv, argIdx, argPos, &b);

           cachedValue = (void *)realloc(cachedValue, sizeof(b));
           cachedValueLength = sizeof(b);
           memcpy(cachedValue, &b, sizeof(b));

           arg->jb = b;
           *size = sizeof(b);
           // status = clSetKernelArg(programContext->kernel, argPos, sizeof(b), &b);
       }
       else if (isLong()) {
           jlong l;
           getPrimitive(jenv, argIdx, argPos, &l);

           cachedValue = (void *)realloc(cachedValue, sizeof(l));
           cachedValueLength = sizeof(l);
           memcpy(cachedValue, &l, sizeof(l));

           arg->jl = l;
           *size = sizeof(l);
           // status = clSetKernelArg(programContext->kernel, argPos, sizeof(l), &l);
       }
       else if (isDouble()) {
           jdouble d;
           getPrimitive(jenv, argIdx, argPos, &d);

           cachedValue = (void *)realloc(cachedValue, sizeof(d));
           cachedValueLength = sizeof(d);
           memcpy(cachedValue, &d, sizeof(d));

           arg->jd = d;
           *size = sizeof(d);
           // status = clSetKernelArg(programContext->kernel, argPos, sizeof(d), &d);
       }
   }
   return status;
}

void KernelArg::dumpTypeToFile(FILE *fp) {
    char buffer[128];
    buffer[0] = '\0';

    if (isFloat()) {
        strcat(buffer, "float");
    } else if (isInt()) {
        strcat(buffer, "int");
    } else if (isBoolean()) {
        strcat(buffer, "bool");
    } else if (isByte()) {
        strcat(buffer, "byte");
    } else if (isLong()) {
        strcat(buffer, "long");
    } else if (isDouble()) {
        strcat(buffer, "double");
    }

    if (isArray()) {
        strcat(buffer, "[]");
    }

    reliableWrite(buffer, sizeof(char), strlen(buffer) + 1, fp);
}

size_t KernelArg::getLengthForType() {
    size_t length = 0;
    if (isFloat() || isInt()) {
        length = 4;
    } else if (isBoolean()) {
        length = sizeof(jboolean);
    } else if (isByte()) {
        length = sizeof(jbyte);
    } else if (isLong() || isDouble()) {
        length = 8;
    }
    return length;
}

void KernelArg::dumpLengthInBytesToFile(FILE *fp, int relaunch, JNIEnv *jenv) {
    if (!isArray()) {
        if (relaunch) {
            reliableWrite(&cachedValueLength, sizeof(size_t), 1, fp);
        } else {
            size_t length = getLengthForType();
            reliableWrite(&length, sizeof(size_t), 1, fp);
        }
    } else {
         if (relaunch == 0) {
             syncSizeInBytes(jenv);
         }
         reliableWrite(&arrayBuffer->lengthInBytes, sizeof(size_t), 1, fp);
    }
}

void KernelArg::dumpData(FILE *fp, int relaunch, JNIEnv *jenv,
        JNIContext *jniContext, OpenCLContext *openclContext) {
    int willDumpData = 1;
    int wontDumpData = 0;

    if (!isArray()) {
        reliableWrite(&willDumpData, sizeof(int), 1, fp);
        if (relaunch) {
            reliableWrite(cachedValue, getLengthForType(), 1, fp);
        } else {
            if (isFloat()) {
                jfloat f;
                getPrimitive(jenv, 0, 0, &f);
                reliableWrite(&f, sizeof(float), 1, fp);
            }
            else if (isInt()) {
                jint i;
                getPrimitive(jenv, 0, 0, &i);
                reliableWrite(&i, sizeof(int), 1, fp);
            }
            else if (isBoolean()) {
                jboolean z;
                getPrimitive(jenv, 0, 0, &z);
                reliableWrite(&z, sizeof(z), 1, fp);
            }
            else if (isByte()) {
                jbyte b;
                getPrimitive(jenv, 0, 0, &b);
                reliableWrite(&b, sizeof(b), 1, fp);
            }
            else if (isLong()) {
                jlong l;
                getPrimitive(jenv, 0, 0, &l);
                reliableWrite(&l, sizeof(l), 1, fp);
            }
            else if (isDouble()) {
                jdouble d;
                getPrimitive(jenv, 0, 0, &d);
                reliableWrite(&d, sizeof(d), 1, fp);
            }
        }
    } else {
        if (relaunch == 0) {
            arrayBuffer->javaArray = (jarray)jenv->GetObjectField(javaArg,
                    javaArrayFieldID);
        }

        if (zeroBeforeKernel) {
            reliableWrite(&willDumpData, sizeof(int), 1, fp);
            char zeroBuf[4096];
            memset(zeroBuf, 0x00, sizeof(char) * 4096);
            int soFar = 0;
            while (soFar < arrayBuffer->lengthInBytes) {
                int toWrite = 4096;
                if (toWrite > arrayBuffer->lengthInBytes - soFar) {
                    toWrite = arrayBuffer->lengthInBytes - soFar;
                }
                reliableWrite(zeroBuf, 1, toWrite, fp);
                soFar += toWrite;
            }
        } else if (dir != OUT) {
            reliableWrite(&willDumpData, sizeof(int), 1, fp);

            if (relaunch) {
                cl_mem mem = jniContext->datactx->hadoopclRefresh(this, relaunch,
                    jniContext, openclContext);
                void *buf = malloc(arrayBuffer->lengthInBytes);
                clEnqueueReadBuffer(openclContext->copyCommandQueue, mem, CL_TRUE,
                    0, arrayBuffer->lengthInBytes, buf, 0, NULL, NULL);
                reliableWrite(buf, getLengthForType(),
                    arrayBuffer->lengthInBytes / getLengthForType(),
                    fp);
                free(buf);
            } else {
                pin(jenv);
                reliableWrite(arrayBuffer->addr, getLengthForType(),
                        arrayBuffer->lengthInBytes / getLengthForType(), fp);
                unpinAbort(jenv);
            }
        } else {
            reliableWrite(&wontDumpData, sizeof(int), 1, fp);
        }
    }
}

void KernelArg::dumpToFile(FILE *fp, int relaunch, JNIEnv *jenv,
        JNIContext *jniContext, OpenCLContext *openclContext) {
    int isRef = 1;
    int isNotRef = 0;

    dumpTypeToFile(fp);
    reliableWrite(name, sizeof(char), strlen(name) + 1, fp);
    dumpLengthInBytesToFile(fp, relaunch, jenv);
    if (isArray()) {
        reliableWrite(&isRef, sizeof(int), 1, fp);
    } else {
        reliableWrite(&isNotRef, sizeof(int), 1, fp);
    }
    dumpData(fp, relaunch, jenv, jniContext, openclContext);

    if (usesArrayLength()) {
        int hasData = 1;
        char lengthNameBuf[128];
        size_t lengthOfLength = sizeof(int);
        if (relaunch == 0) {
            syncJavaArrayLength(jenv);
        }
        sprintf(lengthNameBuf, "%s_length", name);

        reliableWrite("int", 1, 4, fp);
        reliableWrite(lengthNameBuf, 1, strlen(lengthNameBuf) + 1, fp);
        reliableWrite(&lengthOfLength, sizeof(size_t), 1, fp);
        reliableWrite(&isNotRef, sizeof(int), 1, fp);
        reliableWrite(&hasData, sizeof(int), 1, fp);
        reliableWrite(&(arrayBuffer->length), lengthOfLength, 1, fp);
    }
}

