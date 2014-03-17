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
#define TRACE_LINE fprintf(stderr, "%s:%d | task %d attempt %d context %d launch %d\n", __FILE__, __LINE__, jniContext->taskId, jniContext->attemptId, jniContext->contextId, jniContext->kernelLaunchCounter - 1);
#else
#define TRACE_LINE
#endif

//this is a workaround for windows machines since <windows.h> defines min/max that break code.
#define NOMINMAX

#include "Aparapi.h"
#include "Config.h"
#include "ProfileInfo.h"
#include "ArrayBuffer.h"
#include "AparapiBuffer.h"
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

//compiler dependant code
/**
 * calls either clEnqueueMarker or clEnqueueMarkerWithWaitList 
 * depending on the version of OpenCL installed.
 * convenience function so we don't have to have #ifdefs all over the code
 *
 * Actually I backed this out (Gary) when issue #123 was reported.  This involved
 * a build on a 1.2 compatible platform which failed on a platform with a 1.1 runtime. 
 * Failed to link. 
 * The answer is to set   -DCL_USE_DEPRECATED_OPENCL_1_1_APIS at compile time and *not* use 
 * the CL_VERSION_1_2 ifdef.
 */
int enqueueMarker(cl_command_queue commandQueue, cl_event* firstEvent) {
//#ifdef CL_VERSION_1_2
//   return clEnqueueMarkerWithWaitList(commandQueue, 0, NULL, firstEvent);
//#else
   // this was deprecated in 1.1 make sure we use -DCL_USE_DEPRECATED_OPENCL_1_1_APIS
   return clEnqueueMarker(commandQueue, firstEvent);
//#endif
}

/**
 * calls either GetCurrentProcessId or getpid depending on if we're on WIN32 or any other system
 * conveiniece function so we don't have to have #ifdefs all over the code
 */
jint getProcess() {
#if defined (_WIN32)
   return GetCurrentProcessId();
#else
   return (jint)getpid();
#endif
}


JNI_JAVA(jint, KernelRunnerJNI, disposeJNI)
   (JNIEnv *jenv, jobject jobj, jlong jniContextHandle) {
      if (config== NULL){
         config = new Config(jenv);
      }
      cl_int status = CL_SUCCESS;
      JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);
      if (jniContext != NULL){
         jniContext->dispose(jenv, config);
         delete jniContext;
         jniContext = NULL;
      }
      return(status);
   }

/*
void idump(const char *str, void *ptr, int size){
   int * iptr = (int *)ptr;
   for (unsigned i=0; i<size/sizeof(int); i++){
      fprintf(stderr, "%s%4d %d\n", str, i, iptr[i]);
   }
}

void fdump(const char *str, void *ptr, int size){
   float * fptr = (float *)ptr;
   for (unsigned i=0; i<size/sizeof(float); i++){
      fprintf(stderr, "%s%4d %6.2f\n", str, i, fptr[i]);
   }
}
*/


jint writeProfileInfo(JNIContext* jniContext){
   cl_ulong currSampleBaseTime = -1;
   int pos = 1;

   if (jniContext->firstRun) {
      fprintf(jniContext->profileFile, "# PROFILE Name, queued, submit, start, end (microseconds)\n");
   }       

   // A read by a user kernel means the OpenCL layer wrote to the kernel and vice versa
   for (int i=0; i< jniContext->argc; i++){
      KernelArg *arg=jniContext->args[i];
      if (arg->isBackedByArray() && arg->isReadByKernel()){

         // Initialize the base time for this sample
         if (currSampleBaseTime == -1) {
            currSampleBaseTime = arg->arrayBuffer->write.queued;
         } 
         fprintf(jniContext->profileFile, "%d write %s,", pos++, arg->name);

         fprintf(jniContext->profileFile, "%lu,%lu,%lu,%lu,",  
        	(unsigned long)(arg->arrayBuffer->write.queued - currSampleBaseTime)/1000,
        	(unsigned long)(arg->arrayBuffer->write.submit - currSampleBaseTime)/1000,
        	(unsigned long)(arg->arrayBuffer->write.start - currSampleBaseTime)/1000,
        	(unsigned long)(arg->arrayBuffer->write.end - currSampleBaseTime)/1000);
      }
   }

   for (jint pass=0; pass<jniContext->passes; pass++){

      // Initialize the base time for this sample if necessary
      if (currSampleBaseTime == -1) {
         currSampleBaseTime = jniContext->exec[pass].queued;
      } 

      // exec 
      fprintf(jniContext->profileFile, "%d exec[%d],", pos++, pass);

      fprintf(jniContext->profileFile, "%lu,%lu,%lu,%lu,",  
            (unsigned long)(jniContext->exec[pass].queued - currSampleBaseTime)/1000,
            (unsigned long)(jniContext->exec[pass].submit - currSampleBaseTime)/1000,
            (unsigned long)(jniContext->exec[pass].start - currSampleBaseTime)/1000,
            (unsigned long)(jniContext->exec[pass].end - currSampleBaseTime)/1000);
   }

   // 
   if ( jniContext->argc == 0 ) {
      fprintf(jniContext->profileFile, "\n");
   } else { 
      for (int i=0; i< jniContext->argc; i++){
         KernelArg *arg=jniContext->args[i];
         if (arg->isBackedByArray() && arg->isMutableByKernel()){

            // Initialize the base time for this sample
            if (currSampleBaseTime == -1) {
               currSampleBaseTime = arg->arrayBuffer->read.queued;
            }

            fprintf(jniContext->profileFile, "%d read %s,", pos++, arg->name);

            fprintf(jniContext->profileFile, "%lu,%lu,%lu,%lu,",  
            	(unsigned long)(arg->arrayBuffer->read.queued - currSampleBaseTime)/1000,
            	(unsigned long)(arg->arrayBuffer->read.submit - currSampleBaseTime)/1000,
            	(unsigned long)(arg->arrayBuffer->read.start - currSampleBaseTime)/1000,
            	(unsigned long)(arg->arrayBuffer->read.end - currSampleBaseTime)/1000);
         }
      }
   }
   fprintf(jniContext->profileFile, "\n");
   return(0);
}

// Should failed profiling abort the run and return early?
cl_int profile(ProfileInfo *profileInfo, cl_event *event, jint type, char* name, cl_ulong profileBaseTime ) {

   cl_int status = CL_SUCCESS;

   try {
      status = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_QUEUED, sizeof(profileInfo->queued), &(profileInfo->queued), NULL);
      if(status != CL_SUCCESS) throw CLException(status, "clGetEventProfiliningInfo() QUEUED");

      status = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_SUBMIT, sizeof(profileInfo->submit), &(profileInfo->submit), NULL);
      if(status != CL_SUCCESS) throw CLException(status, "clGetEventProfiliningInfo() SUBMIT");

      status = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(profileInfo->start), &(profileInfo->start), NULL);
      if(status != CL_SUCCESS) throw CLException(status, "clGetEventProfiliningInfo() START");

      status = clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(profileInfo->end), &(profileInfo->end), NULL);
      if(status != CL_SUCCESS) throw CLException(status, "clGetEventProfiliningInfo() END");

   } catch(CLException& cle) {
     cle.printError();
     return cle.status();
   }

   profileInfo->queued -= profileBaseTime;
   profileInfo->submit -= profileBaseTime;
   profileInfo->start -= profileBaseTime;
   profileInfo->end -= profileBaseTime;
   profileInfo->type = type;
   profileInfo->name = name;
   profileInfo->valid = true;

   return status;
}


/**
 * Step through all non-primitive (arrays) args
 * and determine if the field has changed
 * The field may have been re-assigned by the Java code to NULL or another instance. 
 * If we detect a change then we discard the previous cl_mem buffer,
 * the caller will detect that the buffers are null and will create new cl_mem buffers. 
 * @param jenv the java environment
 * @param jobj the object we might be updating
 * @param jniContext the context we're working in
 *
 * @throws CLException
 */
jint updateNonPrimitiveReferences(JNIEnv *jenv, jobject jobj, JNIContext* jniContext) {
   cl_int status = CL_SUCCESS;
   if (jniContext != NULL){
      for (jint i = 0; i < jniContext->argc; i++){ 
         KernelArg *arg = jniContext->args[i];

         // make sure that the JNI arg reflects the latest type info from the instance.
         // For example if the buffer is tagged as explicit and needs to be pushed
         arg->syncType(jenv);

         if (config->isVerbose()){
            fprintf(stderr, "got type for %s: %08x\n", arg->name, arg->type);
         }

         //this won't be a problem with the aparapi buffers because
         //we need to copy them every time anyway
         if (!arg->isPrimitive() && !arg->isAparapiBuffer()) {
            // Following used for all primitive arrays, object arrays and nio Buffers
            jarray newRef = (jarray)jenv->GetObjectField(arg->javaArg, KernelArg::javaArrayFieldID);
            if (config->isVerbose()){
               fprintf(stderr, "testing for Resync javaArray %s: old=%p, new=%p\n", arg->name, arg->arrayBuffer->javaArray, newRef);         
            }

            if (!jenv->IsSameObject(newRef, arg->arrayBuffer->javaArray)) {
               if (config->isVerbose()){
                  fprintf(stderr, "Resync javaArray for %s: %p  %p\n", arg->name, newRef, arg->arrayBuffer->javaArray);         
               }
               // Free previous ref if any
               if (arg->arrayBuffer->javaArray != NULL) {
                  jenv->DeleteWeakGlobalRef((jweak) arg->arrayBuffer->javaArray);
                  if (config->isVerbose()){
                     fprintf(stderr, "DeleteWeakGlobalRef for %s: %p\n", arg->name, arg->arrayBuffer->javaArray);         
                  }
               }

               // need to free opencl buffers, run will reallocate later
               if (arg->arrayBuffer->mem != 0) {
                  //fprintf(stderr, "-->releaseMemObject[%d]\n", i);
                  if (config->isTrackingOpenCLResources()){
                     memList.remove(arg->arrayBuffer->mem,__LINE__, __FILE__);
                  }
                  status = clReleaseMemObject((cl_mem)arg->arrayBuffer->mem);
                  //fprintf(stderr, "<--releaseMemObject[%d]\n", i);
                  if(status != CL_SUCCESS) throw CLException(status, "clReleaseMemObject()");
                  arg->arrayBuffer->mem = (cl_mem)0;
               }

               arg->arrayBuffer->addr = NULL;

               // Capture new array ref from the kernel arg object

               if (newRef != NULL) {
                  arg->arrayBuffer->javaArray = (jarray)jenv->NewWeakGlobalRef((jarray)newRef);
                  if (config->isVerbose()){
                     fprintf(stderr, "NewWeakGlobalRef for %s, set to %p\n", arg->name,
                           arg->arrayBuffer->javaArray);         
                  }
               } else {
                  arg->arrayBuffer->javaArray = NULL;
               }

               // Save the lengthInBytes which was set on the java side
               arg->syncSizeInBytes(jenv);

               if (config->isVerbose()){
                  fprintf(stderr, "updateNonPrimitiveReferences, args[%d].lengthInBytes=%llu\n", i, arg->arrayBuffer->lengthInBytes);
               }
            } // object has changed
         }
      } // for each arg
   } // if jniContext != NULL
   return(status);
}

/**
 * manages the memory of KernelArgs that are object.  i.e. handels pinning, and moved objects.
 * currently the only objects supported are arrays.
 *
 * @param jenv the java environment
 * @param jniContext the context we got from java
 * @param arg the argument we're processing
 * @param argPos out: the position of arg in the opencl argument list
 * @param argIdx the position of arg in the argument array
 *
 * @throws CLException
 */
void processObject(JNIEnv* jenv, JNIContext* jniContext, KernelArg* arg, int& argPos, int argIdx) {
    if(arg->isArray()) {
       processArray(jenv, jniContext, arg, argPos, argIdx);
    } else if(arg->isAparapiBuffer()) {
       processBuffer(jenv, jniContext, arg, argPos, argIdx);
    }
}

void processArray(JNIEnv* jenv, JNIContext* jniContext, KernelArg* arg, int& argPos, int argIdx) {

   cl_int status = CL_SUCCESS;

   if (config->isProfilingEnabled()){
      arg->arrayBuffer->read.valid = false;
      arg->arrayBuffer->write.valid = false;
   }

   // pin the arrays so that GC does not move them during the call

   // get the C memory address for the region being transferred
   // this uses different JNI calls for arrays vs. directBufs
   void * prevAddr =  arg->arrayBuffer->addr;
   arg->pin(jenv);

   if (config->isVerbose()) {
      fprintf(stderr, "runKernel: arrayOrBuf ref %p, oldAddr=%p, newAddr=%p, ref.mem=%p isCopy=%s\n",
            arg->arrayBuffer->javaArray, 
            prevAddr,
            arg->arrayBuffer->addr,
            arg->arrayBuffer->mem,
            arg->arrayBuffer->isCopy ? "true" : "false");
      fprintf(stderr, "at memory addr %p, contents: ", arg->arrayBuffer->addr);
      unsigned char *pb = (unsigned char *) arg->arrayBuffer->addr;
      for (int k=0; k<8; k++) {
         fprintf(stderr, "%02x ", pb[k]);
      }
      fprintf(stderr, "\n" );
   }

   // record whether object moved 
   // if we see that isCopy was returned by getPrimitiveArrayCritical, treat that as a move
   bool objectMoved = (arg->arrayBuffer->addr != prevAddr) || arg->arrayBuffer->isCopy;

   if (config->isVerbose()){
      if (arg->isExplicit() && arg->isExplicitWrite()){
         fprintf(stderr, "explicit write of %s\n",  arg->name);
      }
   }

   if (jniContext->firstRun || (arg->arrayBuffer->mem == 0) || objectMoved ){
      if (arg->arrayBuffer->mem != 0 && objectMoved) {
         // we need to release the old buffer 
         if (config->isTrackingOpenCLResources()) {
            memList.remove((cl_mem)arg->arrayBuffer->mem, __LINE__, __FILE__);
         }
         status = clReleaseMemObject((cl_mem)arg->arrayBuffer->mem);
         //fprintf(stdout, "dispose arg %d %0lx\n", i, arg->arrayBuffer->mem);

         //this needs to be reported, but we can still keep going
         CLException::checkCLError(status, "clReleaseMemObject()");

         arg->arrayBuffer->mem = (cl_mem)0;
      }

      updateArray(jenv, jniContext, arg, argPos, argIdx);

   } else {
      // Keep the arg position in sync if no updates were required
      if (arg->usesArrayLength()){
         argPos++;
      }
   }

}

void processBuffer(JNIEnv* jenv, JNIContext* jniContext, KernelArg* arg, int& argPos, int argIdx) {

   cl_int status = CL_SUCCESS;

   if (config->isProfilingEnabled()){
      arg->aparapiBuffer->read.valid = false;
      arg->aparapiBuffer->write.valid = false;
   }

   if (config->isVerbose()) {
      fprintf(stderr, "runKernel: arrayOrBuf addr=%p, ref.mem=%p\n",
            arg->aparapiBuffer->data,
            arg->aparapiBuffer->mem);
      fprintf(stderr, "at memory addr %p, contents: ", arg->aparapiBuffer->data);
      unsigned char *pb = (unsigned char *) arg->aparapiBuffer->data;
      for (int k=0; k<8; k++) {
         fprintf(stderr, "%02x ", pb[k]);
      }
      fprintf(stderr, "\n" );
   }

   if (config->isVerbose()){
      if (arg->isExplicit() && arg->isExplicitWrite()){
         fprintf(stderr, "explicit write of %s\n",  arg->name);
      }
   }

   if (arg->aparapiBuffer->mem != 0) {
      if (config->isTrackingOpenCLResources()) {
         memList.remove((cl_mem)arg->aparapiBuffer->mem, __LINE__, __FILE__);
      }
      status = clReleaseMemObject((cl_mem)arg->aparapiBuffer->mem);
      //fprintf(stdout, "dispose arg %d %0lx\n", i, arg->aparapiBuffer->mem);

      //this needs to be reported, but we can still keep going
      CLException::checkCLError(status, "clReleaseMemObject()");

      arg->aparapiBuffer->mem = (cl_mem)0;
   }

   updateBuffer(jenv, jniContext, arg, argPos, argIdx);

}

/**
 * wait for and release all the read events
 *
 * @param jniContext the context we got from Java
 * @param readEventCount the number of read events to wait for
 * @param passes the number of passes for the kernel
 *
 * @throws CLException
 */
void waitForReadEvents(JNIContext* jniContext, int readEventCount, int passes) {

   // don't change the order here
   // We wait for the reads which each depend on the execution, which depends on the writes ;)
   // So after the reads have completed, we can release the execute and writes.
   
   cl_int status = CL_SUCCESS;

   if (readEventCount > 0){

      status = clWaitForEvents(readEventCount, jniContext->readEvents);
      if (status != CL_SUCCESS) throw CLException(status, "clWaitForEvents() read events");

      for (int i=0; i < readEventCount; i++){

         if (config->isProfilingEnabled()) {

            status = profile(&jniContext->args[jniContext->readEventArgs[i]]->arrayBuffer->read, &jniContext->readEvents[i], 0,jniContext->args[jniContext->readEventArgs[i]]->name, jniContext->profileBaseTime);
            if (status != CL_SUCCESS) throw CLException(status, "");
         }
         status = clReleaseEvent(jniContext->readEvents[i]);
         if (status != CL_SUCCESS) throw CLException(status, "clReleaseEvent() read event");

         if (config->isTrackingOpenCLResources()){
            readEventList.remove(jniContext->readEvents[i],__LINE__, __FILE__);
         }
      }
   } else {
      // if readEventCount == 0 then we don't need any reads so we just wait for the executions to complete
      status = clWaitForEvents(1, jniContext->executeEvents);
      if (status != CL_SUCCESS) throw CLException(status, "clWaitForEvents() execute event");
   }

   if (config->isTrackingOpenCLResources()){
      executeEventList.remove(jniContext->executeEvents[0],__LINE__, __FILE__);
   }
   if (config->isProfilingEnabled()) {
      status = profile(&jniContext->exec[passes-1], &jniContext->executeEvents[0], 1, NULL, jniContext->profileBaseTime); // multi gpu ?
      if (status != CL_SUCCESS) throw CLException(status, "");
   }

}

/**
 * check to make sure opencl exited correctly and update java memory.
 *
 * @param jenv the java environment
 * @param jniContext the context we got from Java
 * @param writeEventCount the number of write events to wait for
 *
 * @throws CLException
 */
void checkEvents(JNIEnv* jenv, JNIContext* jniContext, int writeEventCount) {
   // extract the execution status from the executeEvent
   cl_int status;
   cl_int executeStatus;

   status = clGetEventInfo(jniContext->executeEvents[0], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &executeStatus, NULL);
   if (status != CL_SUCCESS) throw CLException(status, "clGetEventInfo() execute event");
   if (executeStatus != CL_COMPLETE) throw CLException(executeStatus, "Execution status of execute event");

   status = clReleaseEvent(jniContext->executeEvents[0]);
   if (status != CL_SUCCESS) throw CLException(status, "clReleaseEvent() read event");

   for (int i = 0; i < writeEventCount; i++) {

      if (config->isProfilingEnabled()) {
         profile(&jniContext->args[jniContext->writeEventArgs[i]]->arrayBuffer->write, &jniContext->writeEvents[i], 2, jniContext->args[jniContext->writeEventArgs[i]]->name, jniContext->profileBaseTime);
      }

      status = clReleaseEvent(jniContext->writeEvents[i]);
      if (status != CL_SUCCESS) throw CLException(status, "clReleaseEvent() write event");

      if (config->isTrackingOpenCLResources()){
         writeEventList.remove(jniContext->writeEvents[i],__LINE__, __FILE__);
      }
   }

   jniContext->unpinAll(jenv);

   if (config->isProfilingCSVEnabled()) {
      writeProfileInfo(jniContext);
   }
   if (config->isTrackingOpenCLResources()){
      fprintf(stderr, "following execution of kernel{\n");
      commandQueueList.report(stderr);
      memList.report(stderr); 
      readEventList.report(stderr); 
      executeEventList.report(stderr); 
      writeEventList.report(stderr); 
      fprintf(stderr, "}\n");
   }

   jniContext->firstRun = false;
}

static void releaseAllEvents(cl_event *events, int nEvents) {
    int i;
    cl_int err;
    for (i = 0; i < nEvents; i++) {
        err = clReleaseEvent(events[i]);
        if (err != CL_SUCCESS) {
            fprintf(stderr,"Error releasing event %d/%d: %d\n", i+1, nEvents, err);
            exit(6);
        }
    }
    if (events) free(events);
}

static void unpinAll(KernelArg **toUnpin, int nToUnpin, JNIEnv *jenv) {
    for (int i = 0; i < nToUnpin; i++) {
        toUnpin[i]->unpin(jenv);
    }
    if (toUnpin) free(toUnpin);
}

static int contains(char *exts, char *search) {
    int exts_len = strlen(exts);
    int search_len = strlen(search);

    char *iter = exts;
    char *end = exts + exts_len - search_len - 1;

    while (iter <= end) {
        if (strncmp(iter, search, search_len) == 0) {
            return 1;
        }

        while (iter <= end && *iter != ' ') {
            iter++;
        }
        iter++; // seek past space
    }
    return 0;
}

JNI_JAVA(jint, KernelRunnerJNI, hadoopclKernelIsDoneJNI)
    (JNIEnv *jenv, jobject jobj, jlong jniContextHandle) {

    if (config == NULL){
       config = new Config(jenv);
    }
    JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);

    cl_int status;
    cl_int err;

TRACE_LINE
    err = clGetEventInfo(jniContext->exec_event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr,"Error retrieving event info: %d\n",err);
        exit(7);
    }
TRACE_LINE
    return status == CL_COMPLETE;
}

JNI_JAVA(jint, KernelRunnerJNI, hadoopclWaitForKernel)
    (JNIEnv *jenv, jobject jobj, jlong jniContextHandle, jlong openclContextHandle) {
    if (config == NULL){
       config = new Config(jenv);
    }
    JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);
    OpenCLContext *openclContext = ((OpenCLContext*)openclContextHandle);
    OpenCLDataContext *openclDataContext = jniContext->datactx;

    cl_int status;
    cl_int err;

TRACE_LINE

    int willRequireRestart;
    hadoopclParameter *d_willRequireRestart = openclDataContext->findHadoopclParam(
        "memWillRequireRestart");

TRACE_LINE
    err = clWaitForEvents(1, &(jniContext->exec_event));
    if (err != CL_SUCCESS) {
        fprintf(stderr,"Error waiting on exec event: %d\n",err);
        exit(69);
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

JNI_JAVA(jint, KernelRunnerJNI, hadoopclDumpBinary)
    (JNIEnv *jenv, jobject jobj, jstring filename,
     jlong openclProgramContextHandle) {
      cl_int err;
      unsigned char *programBinary = NULL;
      size_t programBinarySize = 0;
      OpenCLProgramContext *programContext =
        ((OpenCLProgramContext *)openclProgramContextHandle);

      const char *filenameChars = jenv->GetStringUTFChars(filename, NULL);
      int string_length = strlen(filenameChars) + 1;
      char *fname = (char *)malloc(string_length);
      memcpy(fname, filenameChars, string_length);
      jenv->ReleaseStringUTFChars(filename, filenameChars);

      cl_program program = programContext->program;

      err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
          sizeof(size_t), &programBinarySize, NULL);
      if (err != CL_SUCCESS) {
          fprintf(stderr, "Error fetching binary size: %d\n", err);
          exit(-1);
      }

      programBinary = (unsigned char *)malloc(programBinarySize);
      err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *),
          &programBinary, NULL);
      if (err != CL_SUCCESS) {
          fprintf(stderr, "Error fetching binary: %d\n", err);
          exit(-1);
      }

      FILE *fp = fopen(fname, "w");
      fwrite(programBinary, 1, programBinarySize, fp);
      fclose(fp);
}

JNI_JAVA(jint, KernelRunnerJNI, hadoopclLaunchKernelJNI)
    (JNIEnv *jenv, jobject jobj, jlong jniContextHandle,
     jlong openclContextHandle, jlong openclProgramContextHandle,
     jobject _range, jint relaunch, jstring label) {

      if (config == NULL){
         config = new Config(jenv);
      }
      JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);
      OpenCLContext *openclContext = ((OpenCLContext *)openclContextHandle);
      OpenCLProgramContext *programContext = ((OpenCLProgramContext *)openclProgramContextHandle);

      Range range(jenv, _range);

      cl_int err = CL_SUCCESS;
      cl_event *fillEvents = (cl_event *)malloc(sizeof(cl_event) * jniContext->argc);
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

      try {
#ifdef PROFILE_HADOOPCL
         jniContext->startWrite = read_timer();
#endif
TRACE_LINE

         pthread_mutex_lock(&programContext->lock);
         int argpos = 0;
         for (int argidx = 0; argidx < jniContext->argc; argidx++, argpos++) {
             KernelArg *arg = jniContext->args[argidx];
#ifdef DUMP_DEBUG
             arg->dumpToFile(dump, relaunch, jenv, jniContext, openclContext);
#endif
             if (!arg->isArray()) {
TRACE_LINE

                 err = arg->setPrimitiveArg(jenv, argidx, argpos,
                         config->isVerbose(), relaunch);
                 if (err != CL_SUCCESS) {
                     fprintf(stderr,"Error setting kernel primitive arg for %d,%s: %d\n",
                             argpos, arg->name, err);
                     exit(8);
                 }
             } else {
                 if (relaunch == 0) {
TRACE_LINE
                     arg->syncSizeInBytes(jenv);
                     arg->arrayBuffer->javaArray = (jarray)jenv->GetObjectField(
                             arg->javaArg, KernelArg::javaArrayFieldID);
                 }
TRACE_LINE

                 cl_mem mem = jniContext->datactx->hadoopclRefresh(arg,
                     relaunch, jniContext, openclContext);

                 if (arg->zeroBeforeKernel) {
TRACE_LINE
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
TRACE_LINE
                   if (relaunch == 0 &&
                           (arg->dir != GLOBAL || jniContext->datactx->writtenAtleastOnce == 0)) {
                       // fprintf(stderr, "Writing argument %s with size %llu\n", arg->name, arg->arrayBuffer->lengthInBytes);
                       arg->pin(jenv);
                       err = clEnqueueWriteBuffer(openclContext->copyCommandQueue, mem,
                               CL_TRUE, 0, arg->arrayBuffer->lengthInBytes,
                               arg->arrayBuffer->addr, 0, NULL, NULL);
                       if (err != CL_SUCCESS) {
                           fprintf(stderr,"Reporting failure of write: %d\n",err);
                           return err;
                       }
                       arg->unpinAbort(jenv);
                   }
                 }
TRACE_LINE

                 err = clSetKernelArg(programContext->kernel, argpos,
                         sizeof(cl_mem), &mem);
                 if (err != CL_SUCCESS) {
                     fprintf(stderr,"Error setting kernel array arg for %d,%s: %d %p\n",
                             argpos, arg->name, err, programContext->kernel);
                     exit(9);
                 }

                 if (arg->usesArrayLength()) {
TRACE_LINE
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

         // -----------
         // fix for Mac OSX CPU driver (and possibly others) 
         // which fail to give correct maximum work group info
         // while using clGetDeviceInfo
         // see: http://www.openwall.com/lists/john-dev/2012/04/10/4
         cl_uint max_group_size[3];
         err = clGetKernelWorkGroupInfo(programContext->kernel,
                                           (cl_device_id)openclContext->deviceId,
                                           CL_KERNEL_WORK_GROUP_SIZE,
                                           sizeof(max_group_size),
                                           &max_group_size, NULL);
         
         if (err != CL_SUCCESS) {
            CLException(err, "clGetKernelWorkGroupInfo()").printError();
         } else {
            range.localDims[0] = std::min((cl_uint)range.localDims[0],
                    max_group_size[0]);
         }


         if (fillEventsSoFar > 0) {
             int i;
             clWaitForEvents(fillEventsSoFar, fillEvents);
             for (i = 0; i < fillEventsSoFar; i++) {
                 clReleaseEvent(fillEvents[i]);
             }
         }
         free(fillEvents);

         jniContext->datactx->writtenAtleastOnce = 1;
         // fprintf(stderr, "running on datactx %p\n", jniContext->datactx);

TRACE_LINE

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
         pthread_mutex_lock(&openclContext->execLock);
         err = clEnqueueNDRangeKernel(
               openclContext->execCommandQueue,
               programContext->kernel,
               range.dims,
               range.offsets,
               range.globalDims,
               range.localDims,
               openclContext->prevExecEvent == NULL ? 0 : 1,
               openclContext->prevExecEvent == NULL ? NULL : &(openclContext->prevExecEvent),
               &(jniContext->exec_event));
         if (err != CL_SUCCESS) {
             pthread_mutex_unlock(&openclContext->execLock);
             fprintf(stderr,"Reporting failure of kernel: %d\n",err);
             return err;
         }
         openclContext->prevExecEvent = jniContext->exec_event;
         pthread_mutex_unlock(&openclContext->execLock);
         pthread_mutex_unlock(&programContext->lock);

         // fprintf(stderr, "kernel in-progress on datactx %p\n", jniContext->datactx);
TRACE_LINE
#ifdef TRACE
         clWaitForEvents(1, &jniContext->exec_event);
#endif
TRACE_LINE

         clFlush(openclContext->execCommandQueue);
      } catch(CLException& cle) {
         cle.printError();
         return cle.status();
      }
      return err;
}

JNI_JAVA(jint, KernelRunnerJNI, hadoopclReadbackJNI)
    (JNIEnv *jenv, jobject jobj, jlong jniContextHandle, jlong openclContextHandle) {
      if (config == NULL){
         config = new Config(jenv);
      }
      JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);
      OpenCLContext *openclContext = ((OpenCLContext*)openclContextHandle);
      // fprintf(stderr, "hadoopclReadbackJNI(jniContext=%p, openclContext=%p)\n", jniContext, openclContext);
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

      try {
#ifdef PROFILE_HADOOPCL
        startRead = read_timer();
#endif
         for (int argidx = 0; argidx < jniContext->argc; argidx++) {
TRACE_LINE
             KernelArg *arg = jniContext->args[argidx];
             if (arg->isArray() && arg->dir != IN && arg->dir != GLOBAL) {
TRACE_LINE
                 arg->syncSizeInBytes(jenv);
                 arg->arrayBuffer->javaArray = (jarray)jenv->GetObjectField(
                         arg->javaArg, KernelArg::javaArrayFieldID);
                 // fprintf(stderr, "name = %s javaArray = %p, javaArrayFieldID = %p, javaArg = %p\n", arg->name, arg->arrayBuffer->javaArray, KernelArg::javaArrayFieldID, arg->javaArg);

                 arg->pin(jenv);

                 // fprintf(stderr, "Reading back argument %s with size %llu\n", arg->name, arg->arrayBuffer->lengthInBytes);

TRACE_LINE
                 cl_mem mem = jniContext->datactx->findHadoopclParam(arg)->allocatedMem;
TRACE_LINE
                 err = clEnqueueReadBuffer(openclContext->copyCommandQueue, mem,
                         CL_TRUE, 0, arg->arrayBuffer->lengthInBytes,
                         arg->arrayBuffer->addr, 1, &(jniContext->exec_event),
                         NULL);

TRACE_LINE
                 arg->unpinCommit(jenv);
                 if (err != CL_SUCCESS) {
                     fprintf(stderr,"Error reading %s of size %llu: %d\n",
                             arg->name, arg->arrayBuffer->lengthInBytes, err);
                     exit(12);
                 }
TRACE_LINE
             }
         }
#ifdef PROFILE_HADOOPCL
        stopRead = read_timer();
#endif
      }
      catch(CLException& cle) {
         cle.printError();
         clReleaseEvent(jniContext->exec_event);
         return cle.status();
      }

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

      pthread_mutex_lock(&openclContext->execLock);
      if (openclContext->prevExecEvent == jniContext->exec_event) {
        openclContext->prevExecEvent = 0;
      }
      pthread_mutex_unlock(&openclContext->execLock);

      clReleaseEvent(jniContext->exec_event);
TRACE_LINE

      return err;
    }

// we return the JNIContext from here 
JNI_JAVA(jlong, KernelRunnerJNI, initJNI)
   (JNIEnv *jenv, jobject jobj, jobject kernelObject, jobject openCLDeviceObject,
    jint flags, jint taskId, jint attemptId, jint contextId) {
      if (openCLDeviceObject == NULL){
         fprintf(stderr, "no device object!\n");
      }
      if (config == NULL){
         config = new Config(jenv);
      }
      cl_int status = CL_SUCCESS;
      JNIContext* jniContext = new JNIContext(jenv, kernelObject, openCLDeviceObject, flags, taskId, attemptId, contextId);

      if (jniContext->isValid()) {

         return((jlong)jniContext);
      } else {
         return(0L);
      }
   }

JNI_JAVA(jlong, KernelRunnerJNI, initOpenCL)
  (JNIEnv *jenv, jclass clazz, jobject _openCLDeviceObject, jint flags, int deviceSlot) {
      cl_int status = CL_SUCCESS;
      if (config == NULL) {
        config = new Config(jenv);
      }
      jobject openCLDeviceObject = jenv->NewGlobalRef(_openCLDeviceObject);

      OpenCLContext *clctx = (OpenCLContext *)malloc(sizeof(OpenCLContext));
      memset(clctx, 0x00, sizeof(OpenCLContext));

      jobject platformInstance = OpenCLDevice::getPlatformInstance(jenv,
              openCLDeviceObject);
      cl_platform_id platformId = OpenCLPlatform::getPlatformId(jenv,
              platformInstance);

      clctx->deviceId = OpenCLDevice::getDeviceId(jenv, openCLDeviceObject, deviceSlot);
      clctx->deviceType = (((
             flags & com_amd_aparapi_internal_jni_KernelRunnerJNI_JNI_FLAG_USE_GPU)
                 == com_amd_aparapi_internal_jni_KernelRunnerJNI_JNI_FLAG_USE_GPU)
                   ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU);

      cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM,
          (cl_context_properties)platformId, 0 };
      cl_context_properties* cprops = (NULL == platformId) ? NULL : cps;
      clctx->context = clCreateContext( cprops, 1, &clctx->deviceId, NULL, NULL,
          &status);
      CLException::checkCLError(status, "clCreateContextFromType()");

      cl_command_queue_properties queue_props = 0;
      // queue_props |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
#ifdef FULLY_PROFILE_HADOOPCL
      queue_props |= CL_QUEUE_PROFILING_ENABLE;
#endif

      clctx->execCommandQueue = clCreateCommandQueue(clctx->context, (cl_device_id)clctx->deviceId,
            queue_props | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE , &status);
      if(status != CL_SUCCESS) throw CLException(status,"clCreateCommandQueue()");
      clctx->copyCommandQueue = clCreateCommandQueue(clctx->context, (cl_device_id)clctx->deviceId,
            queue_props | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &status);
      if(status != CL_SUCCESS) throw CLException(status,"clCreateCommandQueue()");

      pthread_mutex_init(&clctx->execLock, NULL);

      // commandQueueList.add(clctx->execCommandQueue, __LINE__, __FILE__);
      // commandQueueList.add(clctx->copyCommandQueue, __LINE__, __FILE__);

      return ((jlong)clctx);
  }

JNI_JAVA(jlong, KernelRunnerJNI, initOpenCLProgram)
  (JNIEnv *jenv, jclass clazz) {
      if (config == NULL) {
        config = new Config(jenv);
      }

      OpenCLProgramContext *ctx =
        (OpenCLProgramContext *)malloc(sizeof(OpenCLProgramContext));
      memset(ctx, 0x00, sizeof(OpenCLProgramContext));
      pthread_mutex_init(&ctx->lock, NULL);

      return ((jlong)ctx);
  }

JNI_JAVA(jlong, KernelRunnerJNI, initOpenCLData)
  (JNIEnv *jenv, jclass clazz) {
      if (config == NULL) {
          config = new Config(jenv);
      }

      OpenCLDataContext *ctx = (OpenCLDataContext *)malloc(sizeof(OpenCLDataContext));
      memset(ctx, 0x00, sizeof(OpenCLDataContext));

      return ((jlong)ctx);
  }

JNI_JAVA(jint, KernelRunnerJNI, initJNIContextFromOpenCLDataContext)
  (JNIEnv *jenv, jobject jobj, jlong jniContextHandle, jlong dataContextHandle) {
      if (config == NULL) {
          config = new Config(jenv);
      }
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

void writeProfile(JNIEnv* jenv, JNIContext* jniContext) {
   // compute profile filename
   // indicate cpu or gpu
   // timestamp
   // kernel name

   jclass classMethodAccess = jenv->FindClass("java/lang/Class"); 
   jmethodID getNameID = jenv->GetMethodID(classMethodAccess,"getName","()Ljava/lang/String;");
   jstring className = (jstring)jenv->CallObjectMethod(jniContext->kernelClass, getNameID);
   const char *classNameChars = jenv->GetStringUTFChars(className, NULL);

   const size_t TIME_STR_LEN = 200;

   char timeStr[TIME_STR_LEN];
   struct tm *tmp;
   time_t t = time(NULL);
   tmp = localtime(&t);
   if (tmp == NULL) {
      perror("localtime");
   }
   //strftime(timeStr, TIME_STR_LEN, "%F.%H%M%S", tmp);  %F seemed to cause a core dump
   strftime(timeStr, TIME_STR_LEN, "%H%M%S", tmp);

   char* fnameStr = new char[strlen(classNameChars) + strlen(timeStr) + 128];
   jint pid = getProcess();

   //sprintf(fnameStr, "%s.%s.%d.%llx\n", classNameChars, timeStr, pid, jniContext);
   sprintf(fnameStr, "aparapiprof.%s.%d.%p", timeStr, pid, jniContext);
   jenv->ReleaseStringUTFChars(className, classNameChars);

   FILE* profileFile = fopen(fnameStr, "w");
   if (profileFile != NULL) {
      jniContext->profileFile = profileFile;
   } else {
      jniContext->profileFile = stderr;
      fprintf(stderr, "Could not open profile data file %s, reverting to stderr\n", fnameStr);
   }
   delete []fnameStr;
}

JNI_JAVA(jlong, KernelRunnerJNI, buildBinaryProgramJNI)
  (JNIEnv *jenv, jclass clazz, jlong openclContextHandle, jstring filename) {
    OpenCLContext *openclContext = ((OpenCLContext *)openclContextHandle);
    if (openclContext == NULL) return 0;
    OpenCLProgramContext *openclProgramContext =
          (OpenCLProgramContext*)malloc(sizeof(OpenCLProgramContext));
    if (openclProgramContext == NULL) return 0;
    memset(openclProgramContext, 0x00, sizeof(OpenCLProgramContext));
}

JNI_JAVA(jlong, KernelRunnerJNI, buildProgramJNI)
   (JNIEnv *jenv, jclass clazz, jlong openclContextHandle, jstring source) {
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

         if(status != CL_SUCCESS) throw CLException(status, "compile()");

         openclProgramContext->kernel = clCreateKernel(
                 openclProgramContext->program, "run", &status);
         if(status != CL_SUCCESS) throw CLException(status,"clCreateKernel()");
      } catch(CLException& cle) {
         cle.printError();
         return 0;
      }
      
      return((jlong)openclProgramContext);
   }


// this is called once when the arg list is first determined for this kernel
JNI_JAVA(jint, KernelRunnerJNI, setArgsJNI)
   (JNIEnv *jenv, jobject jobj, jlong jniContextHandle,
    jlong programContextHandle, jobjectArray argArray, jint argc) {
      if (config == NULL) {
         config = new Config(jenv);
      }
      JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);
      OpenCLProgramContext *programContext = ((OpenCLProgramContext *)programContextHandle);
      cl_int status = CL_SUCCESS;
      if (jniContext != NULL){
         jniContext->argc = argc;
         jniContext->args = new KernelArg*[jniContext->argc];
         jniContext->firstRun = true;

         // Step through the array of KernelArg's to capture the type data for the Kernel's data members.
         for (jint i = 0; i < jniContext->argc; i++){ 
            jobject argObj = jenv->GetObjectArrayElement(argArray, i);
            KernelArg* arg = jniContext->args[i] = new KernelArg(jenv, jniContext, programContext, argObj);
            if (config->isVerbose()){
               if (arg->isExplicit()){
                  fprintf(stderr, "%s is explicit!\n", arg->name);
               }
            }

            if (config->isVerbose()){
               fprintf(stderr, "in setArgs arg %d %s type %08x\n", i, arg->name, arg->type);
               if (arg->isLocal()){
                  fprintf(stderr, "in setArgs arg %d %s is local\n", i, arg->name);
               }else if (arg->isConstant()){
                  fprintf(stderr, "in setArgs arg %d %s is constant\n", i, arg->name);
               }else{
                  fprintf(stderr, "in setArgs arg %d %s is *not* local\n", i, arg->name);
               }
            }

            //If an error occurred, return early so we report the first problem, not the last
            if (jenv->ExceptionCheck() == JNI_TRUE) {
               jniContext->argc = -1;
               delete[] jniContext->args;
               jniContext->args = NULL;
               jniContext->firstRun = true;
               return (status);
            }

         }
         // we will need an executeEvent buffer for all devices
         jniContext->executeEvents = new cl_event[1];

         // We will need *at most* jniContext->argc read/write events
         jniContext->readEvents = new cl_event[jniContext->argc];
         if (config->isProfilingEnabled()) {
            jniContext->readEventArgs = new jint[jniContext->argc];
         }
         jniContext->writeEvents = new cl_event[jniContext->argc];
         if (config->isProfilingEnabled()) {
            jniContext->writeEventArgs = new jint[jniContext->argc];
         }
      }
      return(status);
   }



JNI_JAVA(jstring, KernelRunnerJNI, getExtensionsJNI)
   (JNIEnv *jenv, jobject jobj, jlong openclContextHandle) {
      if (config == NULL){
         config = new Config(jenv);
      }
      jstring jextensions = NULL;
      OpenCLContext *openclContext = ((OpenCLContext*)openclContextHandle);
      if (openclContext != NULL){
         cl_int status = CL_SUCCESS;
         jextensions = CLHelper::getExtensions(jenv, openclContext->deviceId, &status);
      }
      return jextensions;
   }

/**
 * find the arguement in our list of KernelArgs that matches the array the user asked for
 *
 * @param jenv the java environment
 * @param jniContext the context we're working in
 * @param buffer the array we're looking for
 *
 * @return the KernelArg representing the array
 */
KernelArg* getArgForBuffer(JNIEnv* jenv, JNIContext* jniContext, jobject buffer) {
   KernelArg *returnArg = NULL;

   if (jniContext != NULL){
      for (jint i = 0; returnArg == NULL && i < jniContext->argc; i++){ 
         KernelArg *arg = jniContext->args[i];
         if (arg->isArray()) {
            jboolean isSame = jenv->IsSameObject(buffer, arg->arrayBuffer->javaArray);
            if (isSame){
               if (config->isVerbose()){
                  fprintf(stderr, "matched arg '%s'\n", arg->name);
               }
               returnArg = arg;
            }else{
               if (config->isVerbose()){
                  fprintf(stderr, "unmatched arg '%s'\n", arg->name);
               }
            }
         } else if(arg->isAparapiBuffer()) {
            jboolean isSame = jenv->IsSameObject(buffer, arg->aparapiBuffer->getJavaObject(jenv,arg));
            if (isSame) {
               if (config->isVerbose()) {
                  fprintf(stderr, "matched arg '%s'\n", arg->name);
               }
               returnArg = arg;
            } else {
               if (config->isVerbose()) {
                  fprintf(stderr, "unmatched arg '%s'\n", arg->name);
               }
            }
         }
      }
      if (returnArg == NULL){
         if (config->isVerbose()){
            fprintf(stderr, "attempt to get arg for buffer that does not appear to be referenced from kernel\n");
         }
      }
   }
   return returnArg;
}

JNI_JAVA(jobject, KernelRunnerJNI, getProfileInfoJNI)
   (JNIEnv *jenv, jobject jobj, jlong jniContextHandle) {
      if (config == NULL){
         config = new Config(jenv);
      }
      cl_int status = CL_SUCCESS;
      JNIContext* jniContext = JNIContext::getJNIContext(jniContextHandle);
      jobject returnList = NULL;
      if (jniContext != NULL){
         returnList = JNIHelper::createInstance(jenv, ArrayListClass, VoidReturn );
         if (config->isProfilingEnabled()){

            for (jint i = 0; i < jniContext->argc; i++){ 
               KernelArg *arg = jniContext->args[i];
               if (arg->isArray()){
                  if (arg->isMutableByKernel() && arg->arrayBuffer->write.valid){
                     jobject writeProfileInfo = arg->arrayBuffer->write.createProfileInfoInstance(jenv);
                     JNIHelper::callVoid(jenv, returnList, "add", ArgsBooleanReturn(ObjectClassArg), writeProfileInfo);
                  }
               }
            }

            for (jint pass = 0; pass < jniContext->passes; pass++){
               jobject executeProfileInfo = jniContext->exec[pass].createProfileInfoInstance(jenv);
               JNIHelper::callVoid(jenv, returnList, "add", ArgsBooleanReturn(ObjectClassArg), executeProfileInfo);
            }

            for (jint i = 0; i < jniContext->argc; i++){ 
               KernelArg *arg = jniContext->args[i];
               if (arg->isArray()){
                  if (arg->isReadByKernel() && arg->arrayBuffer->read.valid){
                     jobject readProfileInfo = arg->arrayBuffer->read.createProfileInfoInstance(jenv);
                     JNIHelper::callVoid(jenv, returnList, "add", ArgsBooleanReturn(ObjectClassArg), readProfileInfo);
                  }
               }
            }
         }
      }
      return returnList;
   }

