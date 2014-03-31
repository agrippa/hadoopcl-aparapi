package com.amd.aparapi.device;

import com.amd.aparapi.ProfileInfo;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.annotation.Annotation;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.amd.aparapi.Range;
import com.amd.aparapi.internal.opencl.OpenCLArgDescriptor;
import com.amd.aparapi.internal.opencl.OpenCLKernel;
import com.amd.aparapi.internal.opencl.OpenCLPlatform;
import com.amd.aparapi.internal.opencl.OpenCLProgram;
import com.amd.aparapi.opencl.OpenCL;
import com.amd.aparapi.opencl.OpenCL.Arg;
import com.amd.aparapi.opencl.OpenCL.Constant;
import com.amd.aparapi.opencl.OpenCL.GlobalReadOnly;
import com.amd.aparapi.opencl.OpenCL.GlobalReadWrite;
import com.amd.aparapi.opencl.OpenCL.GlobalWriteOnly;
import com.amd.aparapi.opencl.OpenCL.Kernel;
import com.amd.aparapi.opencl.OpenCL.Local;
import com.amd.aparapi.opencl.OpenCL.Resource;
import com.amd.aparapi.opencl.OpenCL.Source;

public class OpenCLDevice extends Device{

   private final OpenCLPlatform platform;

   private final long deviceId;

   private int maxComputeUnits;

   private long localMemSize;

   private long globalMemSize;

   private long maxMemAllocSize;

   private int isAmd;

   /**
    * Minimal constructor
    * 
    * @param _platform
    * @param _deviceId
    * @param _type
    */
   public OpenCLDevice(OpenCLPlatform _platform, long _deviceId, TYPE _type) {
      platform = _platform;
      deviceId = _deviceId;
      type = _type;
   }

   @Override
   public boolean equals(Object obj) {
      if (obj instanceof OpenCLDevice) {
          return this.deviceId == ((OpenCLDevice)obj).deviceId;
      }
      return false;
   }

   @Override
   public int hashCode() {
      return (int)this.deviceId;
   }

   public OpenCLPlatform getOpenCLPlatform() {
      return platform;
   }

   public boolean isAmd() {
       return isAmd > 0;
   }

   public void setIsAmd(int s) {
       isAmd = s;
   }

   public int getMaxComputeUnits() {
      return maxComputeUnits;
   }

   public void setMaxComputeUnits(int _maxComputeUnits) {
      maxComputeUnits = _maxComputeUnits;
   }

   public long getLocalMemSize() {
      return localMemSize;
   }

   public void setLocalMemSize(long _localMemSize) {
      localMemSize = _localMemSize;
   }

   public long getMaxMemAllocSize() {
      return maxMemAllocSize;
   }

   public void setMaxMemAllocSize(long _maxMemAllocSize) {
      maxMemAllocSize = _maxMemAllocSize;
   }

   public long getGlobalMemSize() {
      return globalMemSize;
   }

   public void setGlobalMemSize(long _globalMemSize) {
      globalMemSize = _globalMemSize;
   }

   void setMaxWorkItemSize(int _dim, int _value) {
      maxWorkItemSize[_dim] = _value;
   }

   public long getDeviceId() {
      return (deviceId);
   }

   public List<OpenCLArgDescriptor> getArgs(Method m) {
      final List<OpenCLArgDescriptor> args = new ArrayList<OpenCLArgDescriptor>();
      final Annotation[][] parameterAnnotations = m.getParameterAnnotations();
      final Class<?>[] parameterTypes = m.getParameterTypes();

      for (int arg = 0; arg < parameterTypes.length; arg++) {
         if (parameterTypes[arg].isAssignableFrom(Range.class)) {

         } else {

            long bits = 0L;
            String name = null;
            for (final Annotation pa : parameterAnnotations[arg]) {
               if (pa instanceof GlobalReadOnly) {
                  name = ((GlobalReadOnly) pa).value();
                  bits |= OpenCLArgDescriptor.ARG_GLOBAL_BIT | OpenCLArgDescriptor.ARG_READONLY_BIT;
               } else if (pa instanceof GlobalWriteOnly) {
                  name = ((GlobalWriteOnly) pa).value();
                  bits |= OpenCLArgDescriptor.ARG_GLOBAL_BIT | OpenCLArgDescriptor.ARG_WRITEONLY_BIT;
               } else if (pa instanceof GlobalReadWrite) {
                  name = ((GlobalReadWrite) pa).value();
                  bits |= OpenCLArgDescriptor.ARG_GLOBAL_BIT | OpenCLArgDescriptor.ARG_READWRITE_BIT;
               } else if (pa instanceof Local) {
                  name = ((Local) pa).value();
                  bits |= OpenCLArgDescriptor.ARG_LOCAL_BIT;
               } else if (pa instanceof Constant) {
                  name = ((Constant) pa).value();
                  bits |= OpenCLArgDescriptor.ARG_CONST_BIT | OpenCLArgDescriptor.ARG_READONLY_BIT;
               } else if (pa instanceof Arg) {
                  name = ((Arg) pa).value();
                  bits |= OpenCLArgDescriptor.ARG_ISARG_BIT;
               }

            }
            if (parameterTypes[arg].isArray()) {
               if (parameterTypes[arg].isAssignableFrom(float[].class)) {
                  bits |= OpenCLArgDescriptor.ARG_FLOAT_BIT | OpenCLArgDescriptor.ARG_ARRAY_BIT;
               } else if (parameterTypes[arg].isAssignableFrom(int[].class)) {
                  bits |= OpenCLArgDescriptor.ARG_INT_BIT | OpenCLArgDescriptor.ARG_ARRAY_BIT;
               } else if (parameterTypes[arg].isAssignableFrom(double[].class)) {
                  bits |= OpenCLArgDescriptor.ARG_DOUBLE_BIT | OpenCLArgDescriptor.ARG_ARRAY_BIT;
               } else if (parameterTypes[arg].isAssignableFrom(byte[].class)) {
                  bits |= OpenCLArgDescriptor.ARG_BYTE_BIT | OpenCLArgDescriptor.ARG_ARRAY_BIT;
               } else if (parameterTypes[arg].isAssignableFrom(short[].class)) {
                  bits |= OpenCLArgDescriptor.ARG_SHORT_BIT | OpenCLArgDescriptor.ARG_ARRAY_BIT;
               } else if (parameterTypes[arg].isAssignableFrom(long[].class)) {
                  bits |= OpenCLArgDescriptor.ARG_LONG_BIT | OpenCLArgDescriptor.ARG_ARRAY_BIT;
               }
            } else if (parameterTypes[arg].isPrimitive()) {
               if (parameterTypes[arg].isAssignableFrom(float.class)) {
                  bits |= OpenCLArgDescriptor.ARG_FLOAT_BIT | OpenCLArgDescriptor.ARG_PRIMITIVE_BIT;
               } else if (parameterTypes[arg].isAssignableFrom(int.class)) {
                  bits |= OpenCLArgDescriptor.ARG_INT_BIT | OpenCLArgDescriptor.ARG_PRIMITIVE_BIT;
               } else if (parameterTypes[arg].isAssignableFrom(double.class)) {
                  bits |= OpenCLArgDescriptor.ARG_DOUBLE_BIT | OpenCLArgDescriptor.ARG_PRIMITIVE_BIT;
               } else if (parameterTypes[arg].isAssignableFrom(byte.class)) {
                  bits |= OpenCLArgDescriptor.ARG_BYTE_BIT | OpenCLArgDescriptor.ARG_PRIMITIVE_BIT;
               } else if (parameterTypes[arg].isAssignableFrom(short.class)) {
                  bits |= OpenCLArgDescriptor.ARG_SHORT_BIT | OpenCLArgDescriptor.ARG_PRIMITIVE_BIT;
               } else if (parameterTypes[arg].isAssignableFrom(long.class)) {
                  bits |= OpenCLArgDescriptor.ARG_LONG_BIT | OpenCLArgDescriptor.ARG_PRIMITIVE_BIT;
               }
            } else {
               System.out.println("OUch!");
            }
            if (name == null) {
               throw new IllegalStateException("no name!");
            }
            final OpenCLArgDescriptor kernelArg = new OpenCLArgDescriptor(name, bits);
            args.add(kernelArg);

         }
      }

      return (args);
   }

   private static boolean isReservedInterfaceMethod(Method _methods) {
      return (   _methods.getName().equals("put")
              || _methods.getName().equals("get")
              || _methods.getName().equals("dispose")
              || _methods.getName().equals("begin")
              || _methods.getName().equals("end")
              || _methods.getName().equals("getProfileInfo"));
   }

   private String streamToString(InputStream _inputStream) {
      final StringBuilder sourceBuilder = new StringBuilder();

      if (_inputStream != null) {

         final BufferedReader reader = new BufferedReader(new InputStreamReader(_inputStream));

         try {
            for (String line = reader.readLine(); line != null; line = reader.readLine()) {
               sourceBuilder.append(line).append("\n");
            }
         } catch (final IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace(); System.out.println("Hello");
         }

         try {
            _inputStream.close();
         } catch (final IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace(); System.out.println("Hello");
         }
      }


      return (sourceBuilder.toString());
   }

   public interface DeviceSelector{
      OpenCLDevice select(OpenCLDevice _device);
   }

   public interface DeviceComparitor{
      OpenCLDevice select(OpenCLDevice _deviceLhs, OpenCLDevice _deviceRhs);
   }

   public static OpenCLDevice select(DeviceSelector _deviceSelector) {
      OpenCLDevice device = null;
      final OpenCLPlatform platform = new OpenCLPlatform(0, null, null, null);

      for (final OpenCLPlatform p : platform.getOpenCLPlatforms()) {
         for (final OpenCLDevice d : p.getOpenCLDevices()) {
            device = _deviceSelector.select(d);
            if (device != null) {
               break;
            }
         }
         if (device != null) {
            break;
         }
      }

      return (device);
   }

   public static OpenCLDevice select(DeviceComparitor _deviceComparitor) {
      OpenCLDevice device = null;
      final OpenCLPlatform platform = new OpenCLPlatform(0, null, null, null);

      for (final OpenCLPlatform p : platform.getOpenCLPlatforms()) {
         for (final OpenCLDevice d : p.getOpenCLDevices()) {
            if (device == null) {
               device = d;
            } else {
               device = _deviceComparitor.select(device, d);
            }
         }
      }

      return (device);
   }

   @Override public String toString() {
      final StringBuilder s = new StringBuilder("{");
      boolean first = true;
      for (final int workItemSize : maxWorkItemSize) {
         if (first) {
            first = false;
         } else {
            s.append(", ");
         }

         s.append(workItemSize);
      }

      s.append("}");

      return ("Device " + deviceId + "\n  type:" + type + "\n  maxComputeUnits=" + maxComputeUnits + "\n  maxWorkItemDimensions="
            + maxWorkItemDimensions + "\n  maxWorkItemSizes=" + s + "\n  maxWorkWorkGroupSize=" + maxWorkGroupSize
            + "\n  globalMemSize=" + globalMemSize + "\n  localMemSize=" + localMemSize);
   }
}
