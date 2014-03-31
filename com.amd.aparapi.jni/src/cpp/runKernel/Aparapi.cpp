/*
   Copyright (c) 2010-2011, Advanced Micro Devices, Inc.
   All rights reserved.

   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
   following conditions are met:

   Redistributions of source code must retain the above copyright notice, this list of conditions and the following
   disclaimer. 

   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided with the distribution. 

   Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission. 

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   If you use the software (in whole or in part), you shall adhere to all applicable U.S., European, and other export
   laws, including but not limited to the U.S. Export Administration Regulations ("EAR"), (15 C.F.R. Sections 730 
   through 774), and E.U. Council Regulation (EC) No 1334/2000 of 22 June 2000.  Further, pursuant to Section 740.6 of
   the EAR, you hereby certify that, except pursuant to a license granted by the United States Department of Commerce
   Bureau of Industry and Security or as otherwise permitted pursuant to a License Exception under the U.S. Export 
   Administration Regulations ("EAR"), you will not (1) export, re-export or release to a national of a country in 
   Country Groups D:1, E:1 or E:2 any restricted technology, software, or source code you receive hereunder, or (2) 
   export to Country Groups D:1, E:1 or E:2 the direct product of such technology or software, if such foreign produced
   direct product is subject to national security controls as identified on the Commerce Control List (currently 
   found in Supplement 1 to Part 774 of EAR).  For the most current Country Group listings, or for additional 
   information about the EAR or your obligations under those regulations, please refer to the U.S. Bureau of Industry
   and Security?s website at http://www.bis.doc.gov/. 
   */

// #define TRACE
// #define DUMP_DEBUG
// #define KEEP_DUMP_FILES
// #define PROFILE_HADOOPCL
// #define FULLY_PROFILE_HADOOPCL

#ifdef TRACE
#define TRACE_LINE fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
// #define TRACE_LINE fprintf(stderr, "%s:%d | task %d attempt %d context %d launch %d\n", __FILE__, __LINE__, jniContext->taskId, jniContext->attemptId, jniContext->contextId, jniContext->kernelLaunchCounter - 1);
#else
#define TRACE_LINE
#endif

//this is a workaround for windows machines since <windows.h> defines min/max that break code.
#define NOMINMAX

#include "Aparapi.h"
#include "ArrayBuffer.h"
#include "CLHelper.h"
#include "List.h"
#include "OpenCLJNI.h"
#include <algorithm>
#include <sys/timeb.h>
#include <pthread.h>

unsigned long read_timer() {
  struct timeb tm;
  ftime(&tm);
  return tm.time * 1000 + tm.millitm;
}

void reliableWrite(const void *ptr, size_t size, size_t count, FILE *fp) {
    size_t soFar = 0;
    while (soFar < count) {
        soFar += fwrite(((char *)ptr) + (soFar * size), size, count - soFar, fp);
    }
}

JNI_JAVA(jint, KernelRunnerJNI, disposeJNI)
   (JNIEnv *jenv, jobject jobj, jlong jniContextHandle) {
TRACE_LINE
      JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);
      if (jniContext != NULL){
         jniContext->dispose(jenv);
         delete jniContext;
         jniContext = NULL;
      }
      return 0;
   }

JNI_JAVA(jint, KernelRunnerJNI, hadoopclWaitForKernel)
    (JNIEnv *jenv, jobject jobj, jlong jniContextHandle, jlong openclContextHandle) {

TRACE_LINE

    JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);
    if (jniContext == NULL) {
        fprintf(stderr, "Invalid jniContext in hadoopclWaitForKernel\n");
        exit(1);
    }
    OpenCLContext *openclContext = ((OpenCLContext*)openclContextHandle);
    if (openclContext == NULL) {
        fprintf(stderr, "Invalid openclContext in hadoopclWaitForKernel\n");
        exit(1);
    }
    OpenCLDataContext *openclDataContext = jniContext->datactx;
    if (openclDataContext == NULL) {
        fprintf(stderr, "Invalid openclDataContext in hadoopclWaitForKernel\n");
        exit(1);
    }

TRACE_LINE

    int willRequireRestart;
    hadoopclParameter *d_willRequireRestart = openclDataContext->findHadoopclParam(
        "memWillRequireRestart");
    if (d_willRequireRestart == NULL) {
        fprintf(stderr, "Error finding param memWillRequireRestart in hadoopclWaitForKernel\n");
        exit(1);
    }
    cl_int err = clWaitForEvents(1, &(jniContext->exec_event));
    if (err != CL_SUCCESS) {
        fprintf(stderr,"Error waiting on exec event: %d\n",err);
        exit(69);
    }
    err = clReleaseEvent(jniContext->exec_event);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error releasing exec_event\n");
        exit(1);
    }
TRACE_LINE
    err = clEnqueueReadBuffer(openclContext->copyCommandQueue,
        d_willRequireRestart->allocatedMem, CL_TRUE, 0, sizeof(int),
        &willRequireRestart, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error copying back willRequireRestart: %d\n", err);
        exit(70);
    }
TRACE_LINE
    return willRequireRestart;
}

static unsigned char *findWithLengthGreaterThan(unsigned char **zeroBuffers,
        int *zeroBuffersLength, int nZeroBuffers, size_t length) {
    int i;
    for (i = 0; i < nZeroBuffers; i++) {
        if (zeroBuffersLength[i] >= length) {
            return zeroBuffers[i];
        }
    }
    return NULL;
}

JNI_JAVA(jint, KernelRunnerJNI, hadoopclLaunchKernelJNI)
    (JNIEnv *jenv, jobject jobj, jlong jniContextHandle,
     jlong openclContextHandle, jlong openclProgramContextHandle,
     jint javaGlobalDim, jint javaLocalDim, jint relaunch, jstring label) {
TRACE_LINE

      size_t globalDim = (size_t)javaGlobalDim;
      size_t localDim = (size_t)javaLocalDim;
      JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);
      if (!jniContext) {
          fprintf(stderr, "Invalid jniContext in hadoopclLaunchKernelJNI\n");
          exit(1);
      }
      OpenCLContext *openclContext = ((OpenCLContext *)openclContextHandle);
      if (!openclContext) {
          fprintf(stderr, "Invalid openclContext in hadoopclLaunchKernelJNI\n");
          exit(1);
      }
      OpenCLProgramContext *programContext = ((OpenCLProgramContext *)openclProgramContextHandle);
      if (!programContext) {
          fprintf(stderr, "Invalid programContext in hadoopclLaunchKernelJNI\n");
          exit(1);
      }

      cl_int err = CL_SUCCESS;
      cl_event *fillEvents = (cl_event *)malloc(sizeof(cl_event) * jniContext->argc);
      if (!fillEvents) {
          fprintf(stderr, "Error allocating fillEvents\n"); exit(1);
      }
      int fillEventsSoFar = 0;

#ifndef CL_API_SUFFIX__VERSION_1_2
      unsigned char **zeroBuffers = NULL;
      int *zeroBuffersLength = NULL;
      int nZeroBuffers = 0;
#endif

#ifdef DUMP_DEBUG

         int thisLaunchId = jniContext->kernelLaunchCounter;
         jniContext->kernelLaunchCounter = jniContext->kernelLaunchCounter + 1;
         int nArgs = jniContext->argc + 1;
         int nArgsUsingLength = 0;
         for (int argidx = 0; argidx < jniContext->argc; argidx++) {
             KernelArg *arg = jniContext->args[argidx];
             if (arg->usesArrayLength()) {
                nArgsUsingLength++;
             }
         }
         nArgs += nArgsUsingLength;

         // char dump_filename[512];
         struct timeb current_time;
         ftime(&current_time);

         sprintf(jniContext->dump_filename, "/tmp/kernel-dump-%d-%d-%d-%d-%llu",
             jniContext->taskId, jniContext->attemptId, jniContext->contextId,
             thisLaunchId, current_time.time * 1000 + current_time.millitm);
         FILE *dump = fopen(jniContext->dump_filename, "w");
         reliableWrite(&nArgs, sizeof(int), 1, dump);
#endif

#ifdef PROFILE_HADOOPCL
         jniContext->startWrite = read_timer();
#endif

         pthread_mutex_lock(&programContext->lock);
         int argpos = 0;
         for (int argidx = 0; argidx < jniContext->argc; argidx++, argpos++) {
             KernelArg *arg = jniContext->args[argidx];
#ifdef DUMP_DEBUG
             arg->dumpToFile(dump, relaunch, jenv, jniContext, openclContext);
#endif
             if (!arg->isArray()) {
                 err = arg->setPrimitiveArg(jenv, argidx, argpos,
                         relaunch);
                 if (err != CL_SUCCESS) {
                     fprintf(stderr,"Error setting kernel primitive arg for %d,%s: %d\n",
                             argpos, arg->name, err);
                     exit(8);
                 }
             } else {
                 if (relaunch == 0) {
                     arg->syncSizeInBytes(jenv);
                     if (arg->arrayBuffer->javaArray != NULL) {
                        jenv->DeleteWeakGlobalRef((jweak)arg->arrayBuffer->javaArray);
                     }
                     arg->arrayBuffer->javaArray = (jarray)jenv->GetObjectField(
                             arg->javaArg, KernelArg::javaArrayFieldID);
                 }

                 cl_mem mem = jniContext->datactx->hadoopclRefresh(arg,
                     relaunch, jniContext, openclContext);

                 if (arg->zeroBeforeKernel) {
                     // fprintf(stderr, "Filling argument %s with size %llu\n", arg->name, arg->arrayBuffer->lengthInBytes);
#ifdef CL_API_SUFFIX__VERSION_1_2
                     int zero = 0;
                     err = clEnqueueFillBuffer(openclContext->copyCommandQueue, mem,
                             &zero, sizeof(zero), 0, arg->arrayBuffer->lengthInBytes,
                             0, NULL, fillEvents + fillEventsSoFar);
#else
                     unsigned char *zeroBuf = findWithLengthGreaterThan(zeroBuffers,
                             zeroBuffersLength, nZeroBuffers,
                             arg->arrayBuffer->lengthInBytes);
                     if (zeroBuf == NULL) {
                         zeroBuf = (unsigned char *)malloc(arg->arrayBuffer->lengthInBytes);
                         memset(zeroBuf, 0x00, arg->arrayBuffer->lengthInBytes);
                         zeroBuffers = (unsigned char **)realloc(zeroBuffers,
                                 sizeof(unsigned char *) * (nZeroBuffers + 1));
                         zeroBuffersLength = (int *)realloc(zeroBuffersLength,
                                 sizeof(int) * (nZeroBuffers + 1));
                         zeroBuffers[nZeroBuffers] = zeroBuf;
                         zeroBuffersLength[nZeroBuffers] = arg->arrayBuffer->lengthInBytes;
                         nZeroBuffers++;
                     }
                     err = clEnqueueWriteBuffer(openclContext->copyCommandQueue,
                             mem, CL_FALSE, 0, arg->arrayBuffer->lengthInBytes,
                             zeroBuf, 0, NULL, fillEvents + fillEventsSoFar);
#endif
                     if (err != CL_SUCCESS) {
                        fprintf(stderr, "Error filling with zeros: %d\n", err);
                        exit(1);
                     }
                     fillEventsSoFar++;
                 } else if (arg->dir != OUT) {
                   if (relaunch == 0 &&
                           (arg->dir != GLOBAL || jniContext->datactx->writtenAtleastOnce == 0)) {
                       // fprintf(stderr, "Writing argument %s with size %llu\n", arg->name, arg->arrayBuffer->lengthInBytes);
                       arg->pin(jenv);
                       err = clEnqueueWriteBuffer(openclContext->copyCommandQueue, mem,
                               CL_TRUE, 0, arg->arrayBuffer->lengthInBytes,
                               arg->arrayBuffer->addr, 0, NULL, NULL);
                       arg->unpinAbort(jenv);
                       if (err != CL_SUCCESS) {
                           fprintf(stderr,"Reporting failure of write: %d\n",err);
                           return err;
                       }
                   }
                 }

                 err = clSetKernelArg(programContext->kernel, argpos,
                         sizeof(cl_mem), &mem);
                 if (err != CL_SUCCESS) {
                     fprintf(stderr,"Error setting kernel array arg for %d,%s: %d %p\n",
                             argpos, arg->name, err, programContext->kernel);
                     exit(9);
                 }

                 if (arg->usesArrayLength()) {
                     argpos++;
                     if (relaunch == 0) {
                         arg->syncJavaArrayLength(jenv);
                     }
                     err = clSetKernelArg(programContext->kernel, argpos,
                             sizeof(jint), &(arg->arrayBuffer->length));
                     if (err != CL_SUCCESS) {
                         fprintf(stderr,"Error setting kernel arg for %d,%s: %d\n",
                                 argpos, arg->name, err);
                         exit(10);
                     }
                 }
             }
         }

#ifdef PROFILE_HADOOPCL
         jniContext->stopWrite = read_timer();
#endif

         int dummy_pass = 0;
#ifdef DUMP_DEBUG
         size_t dummy_pass_len = sizeof(dummy_pass);
         int willWriteData = 1;
         int isNotRef = 0;
         reliableWrite("int", 1, 4, dump);
         reliableWrite("pass", 1, 5, dump);
         reliableWrite(&dummy_pass_len, sizeof(size_t), 1, dump);
         reliableWrite(&isNotRef, sizeof(int), 1, dump);
         reliableWrite(&willWriteData, sizeof(int), 1, dump);
         reliableWrite(&dummy_pass, dummy_pass_len, 1, dump);
#endif
         err = clSetKernelArg(programContext->kernel, argpos, sizeof(int),
                 &dummy_pass);
         if (err != CL_SUCCESS) {
             fprintf(stderr,"Error setting kernel arg for dummy pass\n");
             exit(11);
         }

#ifdef DUMP_DEBUG
         int sourceLength = strlen(programContext->source);
         reliableWrite(&sourceLength, sizeof(sourceLength), 1, dump);
         reliableWrite(programContext->source, 1, sourceLength + 1, dump);
         reliableWrite("DONE", 1, 4, dump);
         fclose(dump);
#endif

         if (fillEventsSoFar > 0) {
             int i;
             clWaitForEvents(fillEventsSoFar, fillEvents);
             for (i = 0; i < fillEventsSoFar; i++) {
                 clReleaseEvent(fillEvents[i]);
             }

             for (i = 0; i < nZeroBuffers; i++) {
                 free(zeroBuffers[i]);
             }
             free(zeroBuffersLength);
             free(zeroBuffers);
         }
         free(fillEvents);

         jniContext->datactx->writtenAtleastOnce = 1;

#if defined PROFILE_HADOOPCL || defined FULLY_PROFILE_HADOOPCL
         const char *labelChars = jenv->GetStringUTFChars(label, NULL);
         int string_length = strlen(labelChars) + 1;
         jniContext->currentLabel = (char *)realloc(jniContext->currentLabel,
             string_length);
         memcpy(jniContext->currentLabel, labelChars, string_length);
         jenv->ReleaseStringUTFChars(label, labelChars);
#endif

#ifdef PROFILE_HADOOPCL
         jniContext->startKernel = read_timer();
#endif
         err = clEnqueueNDRangeKernel(
               openclContext->execCommandQueue,
               programContext->kernel,
               1,
               NULL,
               &globalDim,
               &localDim,
               0, NULL,
               &(jniContext->exec_event));
         if (err != CL_SUCCESS) {
             fprintf(stderr,"Reporting failure of kernel: %d\n",err);
             return err;
         }
         pthread_mutex_unlock(&programContext->lock);

TRACE_LINE
#ifdef TRACE
         clWaitForEvents(1, &jniContext->exec_event);
#endif

         clFlush(openclContext->execCommandQueue);
TRACE_LINE
      return err;
}

JNI_JAVA(jint, KernelRunnerJNI, hadoopclReadbackJNI)
    (JNIEnv *jenv, jobject jobj, jlong jniContextHandle, jlong openclContextHandle) {
TRACE_LINE
      JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);
      if (!jniContext) {
          fprintf(stderr, "Invalid jniContext in hadoopclReadbackJNI\n");
          exit(1);
      }
      OpenCLContext *openclContext = ((OpenCLContext*)openclContextHandle);
      if (!openclContext) {
          fprintf(stderr, "Invalid openclContext in hadoopclReadbackJNI\n");
          exit(1);
      }
#ifdef PROFILE_HADOOPCL
      unsigned long startRead, stopRead;
#endif

      cl_int err = CL_SUCCESS;

#ifdef DUMP_DEBUG
#ifndef KEEP_DUMP_FILES
      if (remove(jniContext->dump_filename) != 0) {
          fprintf(stderr, "Error deleting dump file %s\n", jniContext->dump_filename);
          exit(1);
      }
#endif
#endif

#ifdef PROFILE_HADOOPCL
        startRead = read_timer();
#endif
         for (int argidx = 0; argidx < jniContext->argc; argidx++) {
             KernelArg *arg = jniContext->args[argidx];
             if (arg->isArray() && arg->dir != IN && arg->dir != GLOBAL) {
                 arg->syncSizeInBytes(jenv);
                 if (arg->arrayBuffer->javaArray != NULL) {
                    jenv->DeleteWeakGlobalRef((jweak)arg->arrayBuffer->javaArray);
                 }
                 arg->arrayBuffer->javaArray = (jarray)jenv->GetObjectField(
                         arg->javaArg, KernelArg::javaArrayFieldID);

                 cl_mem mem = jniContext->datactx->findHadoopclParam(arg)->allocatedMem;
                 // fprintf(stderr, "Reading back argument %s with size %llu\n", arg->name, arg->arrayBuffer->lengthInBytes, );

                 arg->pin(jenv);
                 err = clEnqueueReadBuffer(openclContext->copyCommandQueue, mem,
                         CL_TRUE, 0, arg->arrayBuffer->lengthInBytes,
                         arg->arrayBuffer->addr, 0, NULL, NULL);
                 arg->unpinCommit(jenv);
                 if (err != CL_SUCCESS) {
                     fprintf(stderr,"Error reading %s of size %llu: %d\n",
                             arg->name, arg->arrayBuffer->lengthInBytes, err);
                     exit(12);
                 }
             }
         }
#ifdef PROFILE_HADOOPCL
        stopRead = read_timer();
#endif

#ifdef PROFILE_HADOOPCL
      fprintf(stderr, "TIMING | OpenCL Profile: write %lu ms (%lu), kernel %lu ms (%lu), read %lu ms (%lu), label %s\n",
          jniContext->startKernel - jniContext->startWrite, jniContext->startWrite,
          startRead - jniContext->startKernel, jniContext->startKernel,
          stopRead - startRead, startRead,
          jniContext->currentLabel);
#endif
#ifdef FULLY_PROFILE_HADOOPCL
      cl_ulong end, queued, start, submit;
      clGetEventProfilingInfo(jniContext->exec_event,
              CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, NULL);
      clGetEventProfilingInfo(jniContext->exec_event,
              CL_PROFILING_COMMAND_SUBMIT, sizeof(submit), &submit, NULL);
      clGetEventProfilingInfo(jniContext->exec_event,
              CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
      clGetEventProfilingInfo(jniContext->exec_event,
              CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

#ifdef DUMP_DEBUG
#ifdef KEEP_DUMP_FILES
      char profile_filename[512];
      snprintf(profile_filename, 512, "%s.prof", jniContext->dump_filename);
      FILE *fp = fopen(profile_filename, "w");
      fprintf(fp, "OpenCL Profile: kernel queued %llu ms, kernel submitted %llu ms, kernel running %llu ms, label %s\n",
          (submit - queued),
          (start - submit),
          (end - start),
          jniContext->currentLabel);
      fclose(fp);
#endif
#endif
      fprintf(stderr, "  TIMING | OpenCL Profile: kernel queued %lu ms, kernel submitted %lu ms, kernel running %lu ms, label %s\n",
          (submit - queued) / 1000000,
          (start - submit) / 1000000,
          (end - start) / 1000000,
          jniContext->currentLabel);
#endif


TRACE_LINE
      return err;
    }

// we return the JNIContext from here 
JNI_JAVA(jlong, KernelRunnerJNI, initJNI)
   (JNIEnv *jenv, jobject jobj, jobject kernelObject, jobject openCLDeviceObject,
    jint flags, jint taskId, jint attemptId, jint contextId) {
TRACE_LINE
      if (openCLDeviceObject == NULL){
         fprintf(stderr, "no device object!\n");
         exit(1);
      }

      cl_int status = CL_SUCCESS;
      JNIContext* jniContext = new JNIContext(jenv, kernelObject,
              openCLDeviceObject, flags, taskId, attemptId, contextId);
      return((jlong)jniContext);
   }

JNI_JAVA(jlong, KernelRunnerJNI, initOpenCL)
  (JNIEnv *jenv, jclass clazz, jint deviceId, int deviceSlot) {
TRACE_LINE
      cl_int status = CL_SUCCESS;
      int i;

      OpenCLContext *clctx = (OpenCLContext *)malloc(sizeof(OpenCLContext));
      memset(clctx, 0x00, sizeof(OpenCLContext));

      cl_uint nplatforms;
      status = clGetPlatformIDs(0, NULL, &nplatforms);
      if (status != CL_SUCCESS) {
          fprintf(stderr, "Error clGetPlatformIDs %d\n", status);
          exit(1);
      }
      cl_platform_id *platformIds = (cl_platform_id *)malloc(sizeof(cl_platform_id) * nplatforms);
      status = clGetPlatformIDs(nplatforms, platformIds, NULL);
      if (status != CL_SUCCESS) {
          fprintf(stderr, "Error clGetPlatformIDs %d\n", status);
          exit(1);
      }
      int sofar = 0;
      cl_uint ndevices;
      for (i = 0; i < nplatforms; i++) {
          status = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, 0, NULL,
                  &ndevices);
          if (status != CL_SUCCESS) {
              fprintf(stderr, "Error clGetDeviceIDs %d\n", status);
              exit(1);
          }
          if (deviceId < sofar + ndevices) {
              break;
          }
          sofar += ndevices;
      }
      if (i == nplatforms) {
          fprintf(stderr, "Unable to find device id %d\n", deviceId);
          exit(1);
      }
      cl_platform_id platformId = platformIds[i];
      cl_device_id *deviceIds = (cl_device_id *)malloc(sizeof(cl_device_id) * ndevices);
      status = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, ndevices, deviceIds, NULL);
      if (status != CL_SUCCESS) {
          fprintf(stderr, "Error clGetDeviceIDs %d\n", status);
          exit(1);
      }
      clctx->deviceId = deviceIds[deviceId - sofar];

#ifdef CL_API_SUFFIX__VERSION_1_2
      if (deviceSlot != -1) {
         int i;
         size_t partitionPropSize;
         cl_int err = clGetDeviceInfo(clctx->deviceId, CL_DEVICE_PARTITION_PROPERTIES, 0, NULL, &partitionPropSize);
         if (err != CL_SUCCESS) {
             fprintf(stderr, "Error checking partition properties: %d\n", err);
             exit(1);
         }
         cl_device_partition_property *supportedProperties = (cl_device_partition_property *)malloc(partitionPropSize);
         err = clGetDeviceInfo(clctx->deviceId, CL_DEVICE_PARTITION_PROPERTIES, partitionPropSize, supportedProperties, NULL);
         if (err != CL_SUCCESS) {
             fprintf(stderr, "Error fetching supported properties: %d\n", err);
             exit(1);
         }

         int supported = 0;
         for (i = 0; i < partitionPropSize / sizeof(cl_device_partition_property); i++) {
             if (supportedProperties[i] == CL_DEVICE_PARTITION_EQUALLY) {
                 supported = 1;
                 break;
             }
         }
         if (supported) {
             cl_uint nComputeUnits;
             err = clGetDeviceInfo(clctx->deviceId, CL_DEVICE_MAX_COMPUTE_UNITS,
                 sizeof(nComputeUnits), &nComputeUnits, NULL);
             if (err != CL_SUCCESS) {
                 fprintf(stderr, "Error getting number of compute units for subdevice creation: %d\n", err);
                 exit(1);
             }
             cl_device_id *subdevices = (cl_device_id *)malloc(
                 sizeof(cl_device_id) * nComputeUnits);
             cl_device_partition_property part_prop[3] = { CL_DEVICE_PARTITION_EQUALLY, (cl_device_partition_property)1, 0 };
             err = clCreateSubDevices(clctx->deviceId, part_prop, nComputeUnits, subdevices,
                 NULL);
             if (err != CL_SUCCESS) {
                 fprintf(stderr, "Error creating subdevices: %d\n", err);
                 exit(1);
             }
             clctx->deviceId = subdevices[deviceSlot];
             free(subdevices);
         }
      }
#endif

      cl_context_properties cps[] = {
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)platformId,
          0 };
      clctx->context = clCreateContext( cps, 1, &clctx->deviceId, NULL, NULL,
          &status);
      if (status != CL_SUCCESS) {
          fprintf(stderr, "Error clCreateContext %d\n", status);
          exit(1);
      }

      cl_command_queue_properties queue_props = 0;
#ifdef FULLY_PROFILE_HADOOPCL
      queue_props |= CL_QUEUE_PROFILING_ENABLE;
#endif

      clctx->execCommandQueue = clCreateCommandQueue(clctx->context,
              clctx->deviceId, queue_props | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &status);
      if(status != CL_SUCCESS) throw CLException(status,"clCreateCommandQueue()");

      clctx->copyCommandQueue = clCreateCommandQueue(clctx->context,
              clctx->deviceId, queue_props | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &status);
      if(status != CL_SUCCESS) throw CLException(status,"clCreateCommandQueue()");

      free(platformIds);
      free(deviceIds);

      return ((jlong)clctx);
  }

JNI_JAVA(jlong, KernelRunnerJNI, initOpenCLProgram)
  (JNIEnv *jenv, jclass clazz) {
      OpenCLProgramContext *ctx =
        (OpenCLProgramContext *)malloc(sizeof(OpenCLProgramContext));
      memset(ctx, 0x00, sizeof(OpenCLProgramContext));
      pthread_mutex_init(&ctx->lock, NULL);

      return ((jlong)ctx);
  }

JNI_JAVA(jlong, KernelRunnerJNI, initOpenCLData)
  (JNIEnv *jenv, jclass clazz) {
      OpenCLDataContext *ctx = (OpenCLDataContext *)malloc(sizeof(OpenCLDataContext));
      memset(ctx, 0x00, sizeof(OpenCLDataContext));

      return ((jlong)ctx);
  }

JNI_JAVA(jint, KernelRunnerJNI, initJNIContextFromOpenCLDataContext)
  (JNIEnv *jenv, jobject jobj, jlong jniContextHandle, jlong dataContextHandle) {
      OpenCLDataContext *openclDataContext = ((OpenCLDataContext*)dataContextHandle);
      if (openclDataContext == NULL) {
          return 0;
      }

      JNIContext *jniContext = JNIContext::getJNIContext(jniContextHandle);
      if (jniContext == NULL) {
          return 0;
      }
      jniContext->datactx = openclDataContext;
  }

JNI_JAVA(jlong, KernelRunnerJNI, buildProgramJNI)
   (JNIEnv *jenv, jclass clazz, jlong openclContextHandle, jstring source) {
TRACE_LINE
      OpenCLContext *openclContext = ((OpenCLContext*)openclContextHandle);
      if (openclContext == NULL){
         return 0;
      }
      OpenCLProgramContext *openclProgramContext =
          (OpenCLProgramContext*)malloc(sizeof(OpenCLProgramContext));
      if (openclProgramContext == NULL) {
          return 0;
      }
      memset(openclProgramContext, 0x00, sizeof(OpenCLProgramContext));

      try {
         cl_int status = CL_SUCCESS;

#ifdef DUMP_DEBUG
         openclProgramContext->program = CLHelper::compile(jenv,
                 openclContext->context,  1, &openclContext->deviceId, source,
                 NULL, &status, &openclProgramContext->source);
#else
         openclProgramContext->program = CLHelper::compile(jenv,
                 openclContext->context,  1, &openclContext->deviceId, source,
                 NULL, &status, NULL);
#endif
TRACE_LINE

         if(status != CL_SUCCESS) throw CLException(status, "compile()");

         openclProgramContext->kernel = clCreateKernel(
                 openclProgramContext->program, "run", &status);
         if(status != CL_SUCCESS) throw CLException(status,"clCreateKernel()");
      } catch(CLException& cle) {
         cle.printError();
         return 0;
      }
TRACE_LINE
      
      return((jlong)openclProgramContext);
   }


// this is called once when the arg list is first determined for this kernel
JNI_JAVA(jint, KernelRunnerJNI, setArgsJNI)
   (JNIEnv *jenv, jobject jobj, jlong jniContextHandle,
    jlong programContextHandle, jobjectArray argArray, jint argc) {
TRACE_LINE
      JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);
      OpenCLProgramContext *programContext = ((OpenCLProgramContext *)programContextHandle);
      if (!programContext) {
          fprintf(stderr, "Invalid programContext in setArgsJNI\n");
          exit(1);
      }
      cl_int status = CL_SUCCESS;
      if (jniContext != NULL){
         jniContext->argc = argc;
         jniContext->args = (KernelArg **)malloc(sizeof(KernelArg *) * jniContext->argc);

         // Step through the array of KernelArg's to capture the type data for the Kernel's data members.
         for (jint i = 0; i < jniContext->argc; i++) {
TRACE_LINE
            jobject argObj = jenv->GetObjectArrayElement(argArray, i);
            KernelArg* arg = jniContext->args[i] = new KernelArg(jenv,
                    jniContext, programContext, argObj);
            //If an error occurred, return early so we report the first problem, not the last
            if (jenv->ExceptionCheck() == JNI_TRUE) {
               jniContext->argc = -1;
               free(jniContext->args);
               jniContext->args = NULL;
               return (status);
            }
         }
         // we will need an executeEvent buffer for all devices
         // We will need *at most* jniContext->argc read/write events
      }
TRACE_LINE
      return(status);
   }

JNI_JAVA(jstring, KernelRunnerJNI, getExtensionsJNI)
   (JNIEnv *jenv, jobject jobj, jlong openclContextHandle) {
TRACE_LINE
      jstring jextensions = NULL;
      OpenCLContext *openclContext = ((OpenCLContext*)openclContextHandle);
      if (openclContext != NULL){
         cl_int status = CL_SUCCESS;
         jextensions = CLHelper::getExtensions(jenv, openclContext->deviceId, &status);
      }
      return jextensions;
   }
