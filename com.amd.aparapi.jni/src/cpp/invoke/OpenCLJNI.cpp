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
   and Security�s website at http://www.bis.doc.gov/. 
   */

/** @opencljni.cpp */

#define OPENCLJNI_SOURCE
#include "OpenCLJNI.h"
#include "OpenCLArgDescriptor.h"
#include "JavaArgs.h"
#include <iostream>

#include "com_amd_aparapi_internal_jni_OpenCLJNI.h"

jobject OpenCLDevice::getPlatformInstance(JNIEnv *jenv, jobject deviceInstance){ 
   return(JNIHelper::getInstanceField<jobject>(jenv, deviceInstance, "platform",
               OpenCLPlatformClassArg ));
}

cl_device_id OpenCLDevice::getDeviceId(JNIEnv *jenv, jobject deviceInstance,
     int deviceSlot){
   cl_device_id devId = ((cl_device_id)JNIHelper::getInstanceField<jlong>(jenv,
         deviceInstance, "deviceId"));
#ifdef CL_API_SUFFIX__VERSION_1_2
   if (deviceSlot != -1) {
      int i;
      size_t partitionPropSize;
      cl_int err = clGetDeviceInfo(devId, CL_DEVICE_PARTITION_PROPERTIES, 0, NULL, &partitionPropSize);
      if (err != CL_SUCCESS) {
          fprintf(stderr, "Error checking partition properties: %d\n", err);
          exit(1);
      }
      cl_device_partition_property *supportedProperties = (cl_device_partition_property *)malloc(partitionPropSize);
      err = clGetDeviceInfo(devId, CL_DEVICE_PARTITION_PROPERTIES, partitionPropSize, supportedProperties, NULL);
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
          err = clGetDeviceInfo(devId, CL_DEVICE_MAX_COMPUTE_UNITS,
              sizeof(nComputeUnits), &nComputeUnits, NULL);
          if (err != CL_SUCCESS) {
              fprintf(stderr, "Error getting number of compute units for subdevice creation: %d\n", err);
              exit(1);
          }
          cl_device_id *subdevices = (cl_device_id *)malloc(
              sizeof(cl_device_id) * nComputeUnits);
          cl_device_partition_property part_prop[3] = { CL_DEVICE_PARTITION_EQUALLY, (cl_device_partition_property)1, 0 };
          err = clCreateSubDevices(devId, part_prop, nComputeUnits, subdevices,
              NULL);
          if (err != CL_SUCCESS) {
              fprintf(stderr, "Error creating subdevices: %d\n", err);
              exit(1);
          }
          devId = subdevices[deviceSlot];
      }
   }
#endif
   return devId;
}

cl_platform_id OpenCLPlatform::getPlatformId(JNIEnv *jenv, jobject platformInstance){
   return((cl_platform_id)JNIHelper::getInstanceField<jlong>(jenv, platformInstance, "platformId"));
}

jint OpenCLRange::getDims(JNIEnv *jenv, jobject rangeInstance){
   return(JNIHelper::getInstanceField<jint>(jenv, rangeInstance, "dims"));
}

const char* localSize(int i) {
    if(i == 0) return "localSize_0";
    if(i == 1) return "localSize_1";
    if(i == 2) return "localSize_2";
    return "localSize_";
}

const char* globalSize(int i) {
    if(i == 0) return "globalSize_0";
    if(i == 1) return "globalSize_1";
    if(i == 2) return "globalSize_2";
    return "globalSize_";
}

void OpenCLRange::fill(JNIEnv *jenv, jobject rangeInstance, jint dims, size_t* offsets, size_t* globalDims, size_t* localDims) {
   for (int i = 0; i < dims && i < 3; i++) {
      offsets[i] = 0;
      localDims[i] = JNIHelper::getInstanceField<jint>(jenv, rangeInstance, localSize(i));
      globalDims[i] = JNIHelper::getInstanceField<jint>(jenv, rangeInstance, globalSize(i));
   }
}

template<typename jT, typename cl_T>
void putPrimative(JNIEnv* jenv, cl_kernel kernel, jobject arg, jint argIndex) {
   cl_T value = JNIHelper::getInstanceField<jT>(jenv, arg, "value");
   cl_int status = clSetKernelArg(kernel, argIndex, sizeof(value), (void *)&(value));
   if (status != CL_SUCCESS) {
      std::cerr << "error setting " << JNIHelper::getType((jT)0) << " arg " << argIndex 
                << " " <<  value << " " << CLHelper::errString(status) << "!\n";
   }
}

JNI_JAVA(jobject, OpenCLJNI, getPlatforms)
   (JNIEnv *jenv, jobject jobj) {
      jobject platformListInstance = JNIHelper::createInstance(jenv, ArrayListClass, VoidReturn);
      cl_int status = CL_SUCCESS;
      cl_uint platformc;

      status = clGetPlatformIDs(0, NULL, &platformc);
      //fprintf(stderr, "There are %d platforms\n", platformc);
      cl_platform_id* platformIds = new cl_platform_id[platformc];
      status = clGetPlatformIDs(platformc, platformIds, NULL);

      if (status == CL_SUCCESS){
         for (unsigned platformIdx = 0; platformIdx < platformc; ++platformIdx) {
            char platformVersionName[512];
            status = clGetPlatformInfo(platformIds[platformIdx], CL_PLATFORM_VERSION, sizeof(platformVersionName), platformVersionName, NULL);

            // fix this so OpenCL 1.3 or higher will not break!
            if (   !strncmp(platformVersionName, "OpenCL 1.2", 10)
                || !strncmp(platformVersionName, "OpenCL 1.1", 10)
#ifdef __APPLE__
                || !strncmp(platformVersionName, "OpenCL 1.0", 10)
#endif
               ) { 
               char platformVendorName[512];  
               char platformName[512];  
               status = clGetPlatformInfo(platformIds[platformIdx], CL_PLATFORM_VENDOR, sizeof(platformVendorName), platformVendorName, NULL);
               status = clGetPlatformInfo(platformIds[platformIdx], CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
               //fprintf(stderr, "platform vendor    %d %s\n", platformIdx, platformVendorName); 
               //fprintf(stderr, "platform version %d %s\n", platformIdx, platformVersionName); 
               jobject platformInstance = JNIHelper::createInstance(jenv,
                       OpenCLPlatformClass, ArgsVoidReturn(LongArg StringClassArg StringClassArg StringClassArg ), 
                     (jlong)platformIds[platformIdx],
                     jenv->NewStringUTF(platformVersionName), 
                     jenv->NewStringUTF(platformVendorName),
                     jenv->NewStringUTF(platformName)
                     );
               if (!platformInstance) {
                   fprintf(stderr, "Constructed invalid platform instance\n");
                   exit(1);
               }
               JNIHelper::callVoid(jenv, platformListInstance, "add",
                       ArgsBooleanReturn(ObjectClassArg), platformInstance);

               cl_uint deviceIdc;
               cl_device_type requestedDeviceType =CL_DEVICE_TYPE_CPU |CL_DEVICE_TYPE_GPU ;
               status = clGetDeviceIDs(platformIds[platformIdx], requestedDeviceType, 0, NULL, &deviceIdc);
               if (status == CL_SUCCESS && deviceIdc > 0 ){
                  cl_device_id* deviceIds = new cl_device_id[deviceIdc];
                  status = clGetDeviceIDs(platformIds[platformIdx], requestedDeviceType, deviceIdc, deviceIds, NULL);
                  if (status == CL_SUCCESS){
                     for (unsigned deviceIdx = 0; deviceIdx < deviceIdc; deviceIdx++){

                        cl_device_type deviceType;
                        status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_TYPE,  sizeof(deviceType), &deviceType, NULL);
                        jobject deviceTypeEnumInstance = JNIHelper::getStaticFieldObject(jenv, DeviceTypeClass, "UNKNOWN", DeviceTypeClassArg);
                        //fprintf(stderr, "device[%d] CL_DEVICE_TYPE = ", deviceIdx);
                        if (deviceType & CL_DEVICE_TYPE_DEFAULT) {
                           deviceType &= ~CL_DEVICE_TYPE_DEFAULT;
                           //fprintf(stderr, "Default ");
                        }
                        if (deviceType & CL_DEVICE_TYPE_CPU) {
                           deviceType &= ~CL_DEVICE_TYPE_CPU;
                           //fprintf(stderr, "CPU ");
                           deviceTypeEnumInstance = JNIHelper::getStaticFieldObject(jenv, DeviceTypeClass, "CPU", DeviceTypeClassArg);
                        }
                        if (deviceType & CL_DEVICE_TYPE_GPU) {
                           deviceType &= ~CL_DEVICE_TYPE_GPU;
                           //fprintf(stderr, "GPU ");
                           deviceTypeEnumInstance = JNIHelper::getStaticFieldObject(jenv, DeviceTypeClass, "GPU", DeviceTypeClassArg);
                        }
                        if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) {
                           deviceType &= ~CL_DEVICE_TYPE_ACCELERATOR;
                        }
                        //fprintf(stderr, "(0x%llx) ", deviceType);
                        //fprintf(stderr, "\n");


                        //fprintf(stderr, "device type pointer %p", deviceTypeEnumInstance);
                        jobject deviceInstance = JNIHelper::createInstance(jenv, OpenCLDeviceClass, ArgsVoidReturn( OpenCLPlatformClassArg LongArg DeviceTypeClassArg  ),
                              platformInstance, 
                              (jlong)deviceIds[deviceIdx],
                              deviceTypeEnumInstance);
                        if (!deviceInstance) {
                            fprintf(stderr, "Constructed invalid device instance\n");
                            exit(1);
                        }
                        JNIHelper::callVoid(jenv, platformInstance, "addOpenCLDevice", ArgsVoidReturn( OpenCLDeviceClassArg ), deviceInstance);


                        cl_uint maxComputeUnits;
                        status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_MAX_COMPUTE_UNITS,  sizeof(maxComputeUnits), &maxComputeUnits, NULL);
                        //fprintf(stderr, "device[%d] CL_DEVICE_MAX_COMPUTE_UNITS = %u\n", deviceIdx, maxComputeUnits);
                        JNIHelper::callVoid(jenv, deviceInstance, "setMaxComputeUnits", ArgsVoidReturn(IntArg),  maxComputeUnits);

                        char *vendor;
                        size_t vendorLength;
                        status = clGetDeviceInfo(deviceIds[deviceIdx],
                            CL_DEVICE_VENDOR, 0, NULL, &vendorLength);
                        vendor = (char *)malloc(vendorLength + 1);
                        clGetDeviceInfo(deviceIds[deviceIdx],
                            CL_DEVICE_VENDOR, vendorLength, vendor, NULL);
                        vendor[vendorLength] = '\0';
                        if (strstr(vendor, "AMD") != NULL || strstr(vendor, "Micro")) {
                            JNIHelper::callVoid(jenv, deviceInstance, "setIsAmd", ArgsVoidReturn(IntArg), 1);
                        } else {
                            JNIHelper::callVoid(jenv, deviceInstance, "setIsAmd", ArgsVoidReturn(IntArg), 0);
                        }
                        free(vendor);

                        cl_uint maxWorkItemDimensions;
                        status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,  sizeof(maxWorkItemDimensions), &maxWorkItemDimensions, NULL);
                        //fprintf(stderr, "device[%d] CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = %u\n", deviceIdx, maxWorkItemDimensions);
                        JNIHelper::callVoid(jenv, deviceInstance, "setMaxWorkItemDimensions",  ArgsVoidReturn(IntArg),  maxWorkItemDimensions);

                        size_t *maxWorkItemSizes = new size_t[maxWorkItemDimensions];
                        status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_MAX_WORK_ITEM_SIZES,  sizeof(size_t)*maxWorkItemDimensions, maxWorkItemSizes, NULL);

                        for (unsigned dimIdx = 0; dimIdx < maxWorkItemDimensions; dimIdx++){
                           //fprintf(stderr, "device[%d] dim[%d] = %d\n", deviceIdx, dimIdx, maxWorkItemSizes[dimIdx]);
                           JNIHelper::callVoid(jenv, deviceInstance, "setMaxWorkItemSize", ArgsVoidReturn(IntArg IntArg), dimIdx,maxWorkItemSizes[dimIdx]);
                        }

                        size_t maxWorkGroupSize;
                        status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_MAX_WORK_GROUP_SIZE,  sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
                        //fprintf(stderr, "device[%d] CL_DEVICE_MAX_GROUP_SIZE = %u\n", deviceIdx, maxWorkGroupSize);
                        JNIHelper::callVoid(jenv, deviceInstance, "setMaxWorkGroupSize",  ArgsVoidReturn(IntArg),  maxWorkGroupSize);

                        cl_ulong maxMemAllocSize;
                        status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_MAX_MEM_ALLOC_SIZE,  sizeof(maxMemAllocSize), &maxMemAllocSize, NULL);
                        //fprintf(stderr, "device[%d] CL_DEVICE_MAX_MEM_ALLOC_SIZE = %lu\n", deviceIdx, maxMemAllocSize);
                        JNIHelper::callVoid(jenv, deviceInstance, "setMaxMemAllocSize",  ArgsVoidReturn(LongArg),  maxMemAllocSize);

                        cl_ulong globalMemSize;
                        status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_GLOBAL_MEM_SIZE,  sizeof(globalMemSize), &globalMemSize, NULL);
                        //fprintf(stderr, "device[%d] CL_DEVICE_GLOBAL_MEM_SIZE = %lu\n", deviceIdx, globalMemSize);
                        JNIHelper::callVoid(jenv, deviceInstance, "setGlobalMemSize", ArgsVoidReturn(LongArg),  globalMemSize);

                        cl_ulong localMemSize;
                        status = clGetDeviceInfo(deviceIds[deviceIdx], CL_DEVICE_LOCAL_MEM_SIZE,  sizeof(localMemSize), &localMemSize, NULL);
                        //fprintf(stderr, "device[%d] CL_DEVICE_LOCAL_MEM_SIZE = %lu\n", deviceIdx, localMemSize);
                        JNIHelper::callVoid(jenv, deviceInstance, "setLocalMemSize", ArgsVoidReturn(LongArg),  localMemSize);
                     }

                  }
               }
            }
         }
      }

      return (platformListInstance);
   }

