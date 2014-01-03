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

void static dumpTypeToFile(FILE *fp) {
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
        strcat("[]");
    }

    fwrite(buffer, sizeof(char), strlen(buffer) + 1, fp);
}

void static getLengthForType() {
    size_t length = 0;
    if (isFloat() || isInt()) {
        length = 4;
    } else if (isBoolean()) {
        length = sizeof(jboolean);
    } else if (isByte()) {
        length = sizeof(jbyte);
    } else if (isLong() || isDouble()) {
        length = 9;
    }
    return length;
}

void static dumpLengthInBytesToFile(FILE *fp, int relaunch) {
    if (!isArray()) {
        if (relaunch) {
            fwrite(&cachedValueLength, sizeof(size_t), 1, fp);
        } else {
            size_t length = getLengthForType();
            fwrite(&length, sizeof(size_t), 1, fp);
        }
    } else {
         if (relaunch == 0) {
             arg->syncSizeInBytes(jenv);
         }
         fwrite(&arrayBuffer->lengthInBytes, sizeof(size_t), 1, fp);
    }
}

void static dumpData(FILE *fp, int relaunch) {
    if (!isArray()) {
        if (relaunch) {
            fwrite(cachedValue, getLengthForType(), 1, fp);
        } else {
            if (isFloat()) {
                jfloat f;
                getPrimitive(jenv, 0, 0, 0, &f);
                fwrite(&f, sizeof(float), 1, fp);
            }
            else if (isInt()) {
                jint i;
                getPrimitive(jenv, argIdx, argPos, verbose, &i);
                fwrite(&i, sizeof(int), 1, fp);
            }
            else if (isBoolean()) {
                jboolean z;
                getPrimitive(jenv, argIdx, argPos, verbose, &z);
                fwrite(&z, sizeof(z), 1, fp);
            }
            else if (isByte()) {
                jbyte b;
                getPrimitive(jenv, argIdx, argPos, verbose, &b);
                fwrite(&b, sizeof(b), 1, fp);
            }
            else if (isLong()) {
                jlong l;
                getPrimitive(jenv, argIdx, argPos, verbose, &l);
                fwrite(&l, sizeof(l), 1, fp);
            }
            else if (isDouble()) {
                jdouble d;
                getPrimitive(jenv, argIdx, argPos, verbose, &d);
                fwrite(&d, sizeof(d), 1, fp);
            }
        }
    } else {
        if (relaunch == 0) {
            arrayBuffer->javaArray = (jarray)jenv->GetObjectField(javaArg,
                    javaArrayFieldID);
        }

        if (zeroBeforeKernel) {
            char zeroBuf[4096];
            memset(zeroBuf, 0x00, sizeof(char) * 4096);
            int soFar = 0;
            while (soFar < arrayBuffer->lengthInBytes) {
                int toWrite = 4096;
                if (toWrite > arrayBuffer->lengthInBytes - soFar) {
                    toWrite = arrayBuffer->lengthInBytes - soFar;
                }
                fwrite(zeroBuf, 1, toWrite, fp);
                soFar += toWrite;
            }
        } else {
            pin(jenv);
            fwrite(arrayBuffer->addr, getLengthForType(),
                    arrayBuffer->lengthInBytes / getLengthForType(), fp);
            unpinAbort(jenv);
        }
    }
}

void KernelArg::dumpToFile(FILE *fp, int relaunch) {
    dumpTypeToFile(fp);
    fwrite(name, sizeof(char), strlen(name) + 1, fp);
    dumpLengthInBytesToFile(fp, relaunch);
    dumpData(fp, relaunch);
}


