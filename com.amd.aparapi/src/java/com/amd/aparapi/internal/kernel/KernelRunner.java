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
laws, including but not limited to the U.S. Export Administration Regulations ("EAR"), (15 C.F.R. Sections 730 through
774), and E.U. Council Regulation (EC) No 1334/2000 of 22 June 2000.  Further, pursuant to Section 740.6 of the EAR,
you hereby certify that, except pursuant to a license granted by the United States Department of Commerce Bureau of 
Industry and Security or as otherwise permitted pursuant to a License Exception under the U.S. Export Administration 
Regulations ("EAR"), you will not (1) export, re-export or release to a national of a country in Country Groups D:1,
E:1 or E:2 any restricted technology, software, or source code you receive hereunder, or (2) export to Country Groups
D:1, E:1 or E:2 the direct product of such technology or software, if such foreign produced direct product is subject
to national security controls as identified on the Commerce Control List (currently found in Supplement 1 to Part 774
of EAR).  For the most current Country Group listings, or for additional information about the EAR or your obligations
under those regulations, please refer to the U.S. Bureau of Industry and Security's website at http://www.bis.doc.gov/. 

*/
package com.amd.aparapi.internal.kernel;

import java.util.ArrayList;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashSet;
import java.util.List;
import java.util.LinkedList;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Map;
import java.util.HashMap;

import java.io.File;
import java.io.FileInputStream;
import java.io.OutputStream;
import java.io.FileOutputStream;
import java.io.PrintWriter;

import com.amd.aparapi.Config;
import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.Constant;
import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.amd.aparapi.Kernel.KernelState;
import com.amd.aparapi.Kernel.Local;
import com.amd.aparapi.ProfileInfo;
import com.amd.aparapi.Range;
import com.amd.aparapi.device.Device;
import com.amd.aparapi.device.OpenCLDevice;
import com.amd.aparapi.internal.exception.AparapiException;
import com.amd.aparapi.internal.exception.CodeGenException;
import com.amd.aparapi.internal.instruction.InstructionSet.TypeSpec;
import com.amd.aparapi.internal.jni.KernelRunnerJNI;
import com.amd.aparapi.internal.model.ClassModel;
import com.amd.aparapi.internal.model.Entrypoint;
import com.amd.aparapi.internal.util.UnsafeWrapper;
import com.amd.aparapi.internal.writer.KernelWriter;
import com.amd.aparapi.opencl.OpenCL;

/**
 * The class is responsible for executing <code>Kernel</code> implementations. <br/>
 * 
 * The <code>KernelRunner</code> is the real workhorse for Aparapi.  Each <code>Kernel</code> instance creates a single
 * <code>KernelRunner</code> to encapsulate state and to help coordinate interactions between the <code>Kernel</code> 
 * and it's execution logic.<br/>
 * 
 * The <code>KernelRunner</code> is created <i>lazily</i> as a result of calling
 * <code>Kernel.execute()</code>. A this time the <code>ExecutionMode</code> is
 * consulted to determine the default requested mode.  This will dictate how the
 * <code>KernelRunner</code> will attempt to execute the <code>Kernel</code>
 *   
 * @see com.amd.aparapi.Kernel#execute(int _globalSize)
 * 
 * @author gfrost
 *
 */
public class KernelRunner extends KernelRunnerJNI {

   private static final AtomicInteger jniContextCounter = new AtomicInteger(0);
   private static Logger logger = Logger.getLogger(Config.getLoggerName());

   private long jniContextHandle = 0;
   private long myOpenCLDataHandle = 0;
   private long myOpenCLContextHandle = 0;

   /*
    * Stores the cl_device_id, cl_context, cl_command_queue, and prevExecEvent
    * (used to serialize all events for a device) objects for a certain device.
    */
   private static Map<Kernel.TaskType, Long> openclContextHandles =
     new HashMap<Kernel.TaskType, Long>();
   /*
   /*
    * Stores the source code, cl_kernel and cl_program objects for a particular
    * MR task type. Intialized once in buildOpenCLContext and then read-only.
    */
   private static Map<Kernel.TaskType, Long> openclProgramContextHandles =
     new HashMap<Kernel.TaskType, Long>();
   /*
    * Stores a list of data contexts for each kernel type, where each data context
    * stores the OpenCL memory necessary to run a kernel. These are created on
    * demand with initOpenCLData but are recycled after dispose() is called.
    */
   private static Map<Kernel.TaskType, List<Long>> openclDataHandles =
     new HashMap<Kernel.TaskType, List<Long>>();
   /*
    * Stores Entrypoint object for each kernel type. These are recycled.
    */
   private static Map<Kernel.TaskType, List<Entrypoint>> entrypoints =
     new HashMap<Kernel.TaskType, List<Entrypoint>>();
   /*
    * Caches the string for a kernel for re-use.
    */
   private static Map<Kernel.TaskType, KernelAndArgLines> kernelCache =
     new HashMap<Kernel.TaskType, KernelAndArgLines>();

   private static Map<Kernel.TaskType, List<KernelArg[]>> kernelArgs =
     new HashMap<Kernel.TaskType, List<KernelArg[]>>();

   static {
       openclProgramContextHandles.put(Kernel.TaskType.MAPPER, new Long(0L));
       openclProgramContextHandles.put(Kernel.TaskType.COMBINER, new Long(0L));
       openclProgramContextHandles.put(Kernel.TaskType.REDUCER, new Long(0L));

       openclDataHandles.put(Kernel.TaskType.MAPPER, new LinkedList<Long>());
       openclDataHandles.put(Kernel.TaskType.COMBINER, new LinkedList<Long>());
       openclDataHandles.put(Kernel.TaskType.REDUCER, new LinkedList<Long>());

       entrypoints.put(Kernel.TaskType.MAPPER, new LinkedList<Entrypoint>());
       entrypoints.put(Kernel.TaskType.COMBINER, new LinkedList<Entrypoint>());
       entrypoints.put(Kernel.TaskType.REDUCER, new LinkedList<Entrypoint>());

       kernelArgs.put(Kernel.TaskType.MAPPER, new LinkedList<KernelArg[]>());
       kernelArgs.put(Kernel.TaskType.COMBINER, new LinkedList<KernelArg[]>());
       kernelArgs.put(Kernel.TaskType.REDUCER, new LinkedList<KernelArg[]>());
   }

   private static long getOpenCLContext(int deviceId, int deviceSlot,
          Kernel.TaskType taskType) {
       final long openclContextHandle;
       synchronized (openclContextHandles) {
           if (openclContextHandles.containsKey(taskType)) {
             openclContextHandle = openclContextHandles.get(taskType);
           } else {
             openclContextHandle = initOpenCL(deviceId, deviceSlot);
             openclContextHandles.put(taskType, openclContextHandle);
           }
       }
       return openclContextHandle;
   }

   private static void buildOpenCLContext(Kernel.TaskType type, String src,
         long openclContextHandle) {
     synchronized (openclProgramContextHandles) {
       if (openclContextHandle == 0) {
         throw new RuntimeException("Got to building before initialization?");
       }
       if (openclProgramContextHandles.get(type) == 0L) {
           long openclProgramContextHandle = buildProgramJNI(openclContextHandle,
               src);
           if (openclProgramContextHandle == 0L) {
             throw new RuntimeException("Failure building OpenCL program context");
           }
           openclProgramContextHandles.put(type,
               new Long(openclProgramContextHandle));
       }
     }
   }

   private final Kernel kernel;

   private Entrypoint entryPoint = null;
   private Entrypoint entryPointCopy = null;
   private boolean fullyInitialized = false;

   private int argc;
   
   private final ExecutorService threadPool = Executors.newCachedThreadPool();
   /**
    * Create a KernelRunner for a specific Kernel instance.
    * 
    * @param _kernel
    */
   public KernelRunner(Kernel _kernel) {
      kernel = _kernel;

   }

   /**
    * <code>Kernel.dispose()</code> delegates to <code>KernelRunner.dispose()</code> which delegates to <code>disposeJNI()</code> to actually close JNI data structures.<br/>
    * 
    * @see KernelRunnerJNI#disposeJNI()
    */
   public void dispose() {
      if (kernel.getExecutionMode().isOpenCL()) {
         disposeJNI(jniContextHandle);
      }

      if (this.myOpenCLDataHandle != 0) {
          final List<Long> dataHandlesForType =
              openclDataHandles.get(kernel.checkTaskType());
          synchronized (dataHandlesForType) {
              dataHandlesForType.add(this.myOpenCLDataHandle);
          }
          this.myOpenCLDataHandle = 0;
      }

      if (this.entryPoint != null) {
          final List<Entrypoint> entrypointsForType = entrypoints.get(
                  this.kernel.checkTaskType());
          synchronized (entrypointsForType) {
              entrypointsForType.add(this.entryPoint);
              if (this.entryPointCopy != null) {
                  entrypointsForType.add(this.entryPointCopy);
              }
          }
          this.entryPoint = null;
          this.entryPointCopy = null;
      }
      threadPool.shutdownNow();
   }

   private Set<String> capabilitiesSet;

   private long accumulatedExecutionTime = 0;

   private long conversionTime = 0;

   private long executionTime = 0;

   /*
   private static Boolean support64BitFloat = null;

   public static boolean allDevicesSupport64BitFloatingPoint() {
       if (support64BitFloat == null) {
           support64BitFloat = new Boolean(allDevicesSupport64bitFloat() > 0);
       }
       return support64BitFloat.booleanValue();
   }
   */

   boolean hasFP64Support() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return (capabilitiesSet.contains(OpenCL.CL_KHR_FP64));
   }

   boolean hasAMDFP64Support() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return (capabilitiesSet.contains(OpenCL.CL_AMD_FP64));
   }

   boolean hasSelectFPRoundingModeSupport() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return capabilitiesSet.contains(OpenCL.CL_KHR_SELECT_FPROUNDING_MODE);
   }

   boolean hasGlobalInt32BaseAtomicsSupport() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return capabilitiesSet.contains(OpenCL.CL_KHR_GLOBAL_INT32_BASE_ATOMICS);
   }

   boolean hasGlobalInt32ExtendedAtomicsSupport() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return capabilitiesSet.contains(OpenCL.CL_KHR_GLOBAL_INT32_EXTENDED_ATOMICS);
   }

   boolean hasLocalInt32BaseAtomicsSupport() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return capabilitiesSet.contains(OpenCL.CL_KHR_LOCAL_INT32_BASE_ATOMICS);
   }

   boolean hasLocalInt32ExtendedAtomicsSupport() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return capabilitiesSet.contains(OpenCL.CL_KHR_LOCAL_INT32_EXTENDED_ATOMICS);
   }

   boolean hasInt64BaseAtomicsSupport() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return capabilitiesSet.contains(OpenCL.CL_KHR_INT64_BASE_ATOMICS);
   }

   boolean hasInt64ExtendedAtomicsSupport() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return capabilitiesSet.contains(OpenCL.CL_KHR_INT64_EXTENDED_ATOMICS);
   }

   boolean has3DImageWritesSupport() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return capabilitiesSet.contains(OpenCL.CL_KHR_3D_IMAGE_WRITES);
   }

   boolean hasByteAddressableStoreSupport() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return capabilitiesSet.contains(OpenCL.CL_KHR_BYTE_ADDRESSABLE_SUPPORT);
   }

   boolean hasFP16Support() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return capabilitiesSet.contains(OpenCL.CL_KHR_FP16);
   }

   boolean hasGLSharingSupport() {
      if (capabilitiesSet == null) {
         throw new IllegalStateException("Capabilities queried before they were initialized");
      }
      return capabilitiesSet.contains(OpenCL.CL_KHR_GL_SHARING);
   }

   private KernelArg[] args = null;

   private void updateKernelArrayRefs() throws AparapiException {
      for (int i = 0; i < argc; i++) {
         final KernelArg arg = args[i];
         try {
            if ((arg.getType() & ARG_ARRAY) != 0) {
               final Object newArrayRef = arg.getField().get(kernel);

               // set up JNI fields for normal arrays
               if (newArrayRef == null) {
                 arg.setNumElements(kernel.getArrayLength(arg.getName()));
               } else {
                 arg.setNumElements(Array.getLength(newArrayRef));

                 if (((args[i].getType() & ARG_EXPLICIT) != 0) &&
                      puts.contains(newArrayRef)) {
                    args[i].setType(args[i].getType() | ARG_EXPLICIT_WRITE);
                    puts.remove(newArrayRef);
                 }
               }
               arg.setSizeInBytes((long)arg.getNumElements() *
                   (long)arg.getPrimitiveSize());
               arg.setJavaArray(newArrayRef);
            }
         } catch (final IllegalArgumentException e) {
            e.printStackTrace();
         } catch (final IllegalAccessException e) {
            e.printStackTrace();
         }
      }
   }

   private Kernel executeOpenCL(final String _entrypointName,
           final Range _range, final int _passes, final boolean enableStriding,
           boolean isRelaunch, String label) throws AparapiException {
      // Read the array refs after kernel may have changed them
      // We need to do this as input to computing the localSize
      assert args != null : "args should not be null";

      updateKernelArrayRefs();

      // native side will reallocate array buffers if necessary
      int execID;
      if ((execID = hadoopclLaunchKernelJNI(jniContextHandle,
              myOpenCLContextHandle,
              openclProgramContextHandles.get(kernel.checkTaskType()),
              _range.getGlobalSize(), _range.getLocalSize(),
              isRelaunch ? 1 : 0, label)) != 0) {
         return null;
      }

      if (logger.isLoggable(Level.FINE)) {
         logger.fine("executeOpenCL completed. " + _range);
      }
      return kernel;
   }

   public synchronized int waitForKernelCompletion() {
     return hadoopclWaitForKernel(jniContextHandle, myOpenCLContextHandle);
   }

   public synchronized int readFromOpenCL() {
       try {
           updateKernelArrayRefs();
       } catch (Exception e) {
           throw new RuntimeException(e);
       }

       return hadoopclReadbackJNI(jniContextHandle, myOpenCLContextHandle);
   }

   private static String readOpenCLAndArgs(String kernelFile, List<String> argsOut) {
        String fileContents;
        try {
            File f = new File(kernelFile);
            FileInputStream fs = new FileInputStream(f);
            byte[] bytes = new byte[(int)f.length()];
            fs.read(bytes);
            fs.close();
            fileContents = new String(bytes, "UTF-8");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        String[] lines = fileContents.split("\n");

        int i = 0;
        while (lines[i].length() > 0) {
            argsOut.add(lines[i]);
            i++;
        }

        i++;
        StringBuilder openclBuilder = new StringBuilder();
        for ( ; i < lines.length; i++) {
            openclBuilder.append(lines[i]);
            openclBuilder.append("\n");
        }
        return openclBuilder.toString();
   }

   private static KernelArg[] constructKernelArgObjects(List<String> argLines, Kernel kernel) {
        KernelArg[] readArgs = new KernelArg[argLines.size()];
        for (int i = 0; i < readArgs.length; i++) {
            String line = argLines.get(i);
            String[] tokens = line.split(" ");
            readArgs[i] = new KernelArg();
            readArgs[i].setName(tokens[0]);
            readArgs[i].setType(Integer.parseInt(tokens[1]));
            readArgs[i].setPrimitiveSize(Integer.parseInt(tokens[2]));
            readArgs[i].setSizeInBytes(Integer.parseInt(tokens[3]));
            readArgs[i].setNumElements(Integer.parseInt(tokens[4]));
            int nDims = Integer.parseInt(tokens[5]);
            int[] dims = new int[nDims];
            for (int j = 7; j < 7 + nDims; j++) {
                dims[j-7] = Integer.parseInt(tokens[j]);
            }
            readArgs[i].setNumDims(nDims);
            readArgs[i].setDims(dims);

            Field f;
            try {
                f = Entrypoint.getFieldFromClassHierarchy(kernel.getClass(), readArgs[i].getName());
            } catch (AparapiException a) {
                throw new RuntimeException(a);
            }

            if (f == null) {
                throw new RuntimeException("Failed to retrieve field for "+readArgs[i].getName());
            }
            readArgs[i].setField(f);
        }
        return readArgs;
   }

   public static void doKernelAndArgLinesPrealloc(Kernel kernel,
         Kernel.TaskType type, int nArgsToPrealloc, OpenCLDevice dev,
         int deviceId, int deviceSlot) {
      if (kernel.getKernelFileForDeviceType(dev.getType()) != null) {
         List<String> argLines = new LinkedList<String>();
         String openCL = readOpenCLAndArgs(kernel.getKernelFileForDeviceType(dev.getType()), argLines);
         synchronized(kernelCache) {
             kernelCache.put(type,
                 new KernelAndArgLines(openCL, argLines));
         }

         List<KernelArg[]> kernelArgsForType = kernelArgs.get(type);
         synchronized (kernelArgsForType) {
            for (int i = 0; i < nArgsToPrealloc; i++) {
                kernelArgsForType.add(constructKernelArgObjects(argLines, kernel));
            }
         }
         buildOpenCLContext(type, openCL, getOpenCLContext(deviceId, deviceSlot,
               type));
      }
   }

   public synchronized void doEntrypointInit(String _entrypointName,
           boolean enableStrided, Device device, int deviceId, int deviceSlot,
           boolean dryRun, int taskId, int attemptId) {
      if (entryPoint == null &&
            (this.kernel.getKernelFileForDeviceType(device.getType()) == null || dryRun)) {

         int requiredNEntrypoints = enableStrided ? 2 : 1;
         final List<Entrypoint> entrypointsForType;
         if (dryRun) {
              entrypointsForType = new ArrayList<Entrypoint>(0);
         } else {
              entrypointsForType = entrypoints.get(
                  this.kernel.checkTaskType());
         }
         final List<Entrypoint> collectedEntrypoints = new ArrayList<Entrypoint>(
                 requiredNEntrypoints);
         synchronized (entrypointsForType) {
             /*
              * Loop while trying to get requiredNEntrypoints entry point
              * objects. These are either created fresh using getEntrypoint or
              * retrieved from entrypointsForType if previously created and
              * disposed().
              */
             ClassModel classModel = null;
             while (requiredNEntrypoints > 0) {
                 if (entrypointsForType.isEmpty()) {
                     // Create new Entrypoint object
                     try {
                        if (classModel == null) {
                            classModel = new ClassModel(kernel.getClass());
                        }
                        Entrypoint created = classModel.getEntrypoint(
                                _entrypointName, kernel);
                        collectedEntrypoints.add(created);

                     } catch (final Exception exception) {
                        throw new RuntimeException(exception);
                     }
                 } else {
                     // Get previously created Entrypoint object
                     collectedEntrypoints.add(entrypointsForType.remove(0));
                 }
                 requiredNEntrypoints--;
             }
         }
         this.entryPoint = collectedEntrypoints.get(0);
         if (enableStrided) {
             this.entryPointCopy = collectedEntrypoints.get(1);
         }
      }

      if (!fullyInitialized) {
         fullyInitialized = true;
         OpenCLDevice openCLDevice = null;
         synchronized (Kernel.class) { // This seems to be needed because of a race condition uncovered with issue #68 http://code.google.com/p/aparapi/issues/detail?id=68
            if (device != null && !(device instanceof OpenCLDevice)) {
               throw new IllegalStateException("range's device is not suitable for OpenCL ");
            }

            openCLDevice = (OpenCLDevice) device; // still might be null! 

            if (dryRun && openCLDevice == null) {
                if (kernel.getExecutionMode().equals(EXECUTION_MODE.GPU)) {
                    openCLDevice = (OpenCLDevice) OpenCLDevice.best();
                } else {
                    openCLDevice = (OpenCLDevice) OpenCLDevice.firstCPU();
                }
            }

            if (openCLDevice == null) {
              throw new RuntimeException("No HadoopCL device specified");
            }

            if (this.myOpenCLContextHandle == 0) {
                this.myOpenCLContextHandle = getOpenCLContext(deviceId,
                    deviceSlot, kernel.checkTaskType());
            }
            // openCLDevice will not be null here
            jniContextHandle = initJNI(kernel, taskId, attemptId,
                jniContextCounter.getAndIncrement());
         } // end of synchronized! issue 68

         if (jniContextHandle == 0) {
            throw new RuntimeException("initJNI failed to return a valid handle");
         }

         if (this.kernel.getKernelFileForDeviceType(device.getType()) == null || dryRun) {
             final String extensions = getExtensionsJNI(myOpenCLContextHandle);
             capabilitiesSet = new HashSet<String>();

             final StringTokenizer strTok = new StringTokenizer(extensions);
             while (strTok.hasMoreTokens()) {
                capabilitiesSet.add(strTok.nextToken());
             }

             if (logger.isLoggable(Level.FINE)) {
                logger.fine("Capabilities initialized to :" + capabilitiesSet.toString());
             }

             if (entryPoint.requiresDoublePragma() && !hasFP64Support() && !hasAMDFP64Support()) {
                throw new RuntimeException("FP64 required but not supported");
             }

             if (entryPoint.requiresByteAddressableStorePragma() && !hasByteAddressableStoreSupport()) {
                throw new RuntimeException("Byte addressable stores required but not supported");
             }

             final boolean all32AtomicsAvailable = hasGlobalInt32BaseAtomicsSupport()
                   && hasGlobalInt32ExtendedAtomicsSupport() && hasLocalInt32BaseAtomicsSupport()
                   && hasLocalInt32ExtendedAtomicsSupport();

             if (entryPoint.requiresAtomic32Pragma() && !all32AtomicsAvailable) {
                throw new RuntimeException("32 bit Atomics required but not supported");
             }
         }

         String openCL = null;
         KernelArg[] readArgs = null;

         if (this.kernel.getKernelFileForDeviceType(device.getType()) != null && !dryRun) {
             List<String> argLines;
             synchronized(kernelCache) {
                 if (kernelCache.containsKey(kernel.checkTaskType())) {
                     KernelAndArgLines forThisKernel =
                       kernelCache.get(kernel.checkTaskType());
                     openCL = forThisKernel.kernel;
                     argLines = forThisKernel.argLines;
                 } else {
                     argLines = new LinkedList<String>();
                     openCL = readOpenCLAndArgs(this.kernel.getKernelFileForDeviceType(device.getType()), argLines);
                     kernelCache.put(kernel.checkTaskType(),
                         new KernelAndArgLines(openCL, argLines));
                 }
             }

             List<KernelArg[]> kernelArgsForType = kernelArgs.get(kernel.checkTaskType());
             synchronized (kernelArgsForType) {
                 if (!kernelArgsForType.isEmpty()) {
                    readArgs = kernelArgsForType.remove(0);
                 } else {
                     readArgs = constructKernelArgObjects(argLines, this.kernel);
                 }
             }
         } else {
             try {
                 synchronized (kernelCache) {
                    if (kernelCache.containsKey(kernel.checkTaskType())) {
                        KernelAndArgLines forThisKernel = kernelCache.get(
                                kernel.checkTaskType());
                        openCL = forThisKernel.kernel;
                    } else {
                        openCL = KernelWriter.writeToString(entryPoint,
                                entryPointCopy,
                                openCLDevice.getType() == Device.TYPE.GPU,
                                enableStrided, hasFP64Support(),
                                hasAMDFP64Support(), openCLDevice.getType() == Device.TYPE.GPU && openCLDevice.isAmd());
                        kernelCache.put(kernel.checkTaskType(),
                                new KernelAndArgLines(openCL, null));
                    }
                 }
             } catch (final CodeGenException codeGenException) {
                throw new RuntimeException(codeGenException);
             }
         }

         if (Config.enableShowGeneratedOpenCL) {
            System.out.println(openCL);
         }

         if (readArgs != null) {
            args = readArgs;
            argc = readArgs.length;
         } else {
             args = new KernelArg[entryPoint.getReferencedFields().size()];

             int i = 0;
             for (final Field field : entryPoint.getReferencedFields()) {
                try {
                   field.setAccessible(true);
                   args[i] = new KernelArg();
                   args[i].setName(field.getName());
                   args[i].setField(field);
                   if ((field.getModifiers() & Modifier.STATIC) == Modifier.STATIC) {
                      args[i].setType(args[i].getType() | ARG_STATIC);
                   }

                   final Class<?> type = field.getType();
                   if (type.isArray()) {
                       args[i].setType(args[i].getType() | ARG_ARRAY);

                       args[i].setType(args[i].getType() | (type.isAssignableFrom(float[].class) ? ARG_FLOAT : 0));
                       args[i].setType(args[i].getType() | (type.isAssignableFrom(int[].class) ? ARG_INT : 0));
                       args[i].setType(args[i].getType() | (type.isAssignableFrom(boolean[].class) ? ARG_BOOLEAN : 0));
                       args[i].setType(args[i].getType() | (type.isAssignableFrom(byte[].class) ? ARG_BYTE : 0));
                       args[i].setType(args[i].getType() | (type.isAssignableFrom(char[].class) ? ARG_CHAR : 0));
                       args[i].setType(args[i].getType() | (type.isAssignableFrom(double[].class) ? ARG_DOUBLE : 0));
                       args[i].setType(args[i].getType() | (type.isAssignableFrom(long[].class) ? ARG_LONG : 0));
                       args[i].setType(args[i].getType() | (type.isAssignableFrom(short[].class) ? ARG_SHORT : 0));

                       // arrays whose length is used will have an int arg holding
                       // the length as a kernel param
                       if (entryPoint.getArrayFieldArrayLengthUsed().contains(args[i].getName())) {
                          args[i].setType(args[i].getType() | ARG_ARRAYLENGTH);
                       }
                   } else if (type.isAssignableFrom(float.class)) {
                      args[i].setType(args[i].getType() | ARG_PRIMITIVE);
                      args[i].setType(args[i].getType() | ARG_FLOAT);
                   } else if (type.isAssignableFrom(int.class)) {
                      args[i].setType(args[i].getType() | ARG_PRIMITIVE);
                      args[i].setType(args[i].getType() | ARG_INT);
                   } else if (type.isAssignableFrom(double.class)) {
                      args[i].setType(args[i].getType() | ARG_PRIMITIVE);
                      args[i].setType(args[i].getType() | ARG_DOUBLE);
                   } else if (type.isAssignableFrom(long.class)) {
                      args[i].setType(args[i].getType() | ARG_PRIMITIVE);
                      args[i].setType(args[i].getType() | ARG_LONG);
                   } else if (type.isAssignableFrom(boolean.class)) {
                      args[i].setType(args[i].getType() | ARG_PRIMITIVE);
                      args[i].setType(args[i].getType() | ARG_BOOLEAN);
                   } else if (type.isAssignableFrom(byte.class)) {
                      args[i].setType(args[i].getType() | ARG_PRIMITIVE);
                      args[i].setType(args[i].getType() | ARG_BYTE);
                   } else if (type.isAssignableFrom(char.class)) {
                      args[i].setType(args[i].getType() | ARG_PRIMITIVE);
                      args[i].setType(args[i].getType() | ARG_CHAR);
                   } else if (type.isAssignableFrom(short.class)) {
                      args[i].setType(args[i].getType() | ARG_PRIMITIVE);
                      args[i].setType(args[i].getType() | ARG_SHORT);
                   }
                } catch (final IllegalArgumentException e) {
                  throw new RuntimeException(e);
                }

                args[i].setPrimitiveSize(getPrimitiveSize(args[i].getType()));

                if (logger.isLoggable(Level.FINE)) {
                   logger.fine("arg " + i + ", " + args[i].getName() + ", type=" + Integer.toHexString(args[i].getType())
                         + ", primitiveSize=" + args[i].getPrimitiveSize());
                }

                i++;
             }
             argc = i;
         }

         if (dryRun) {
            try {
                PrintWriter out = new PrintWriter("fields.dump");
                for (int k = 0; k < args.length; k++) {
                    KernelArg arg = args[k];

                    out.write(arg.getName()+" "+arg.getType()+" "+
                        arg.getPrimitiveSize()+" "+arg.getSizeInBytes()+" "+
                        arg.getNumElements()+" "+arg.getNumDims()+" { ");
                    for (int j = 0; j < arg.getNumDims(); j++) {
                        out.write(arg.getDims()[j]+" ");
                    }
                    out.write("}\n");
                }
                out.write("\n");
                out.write(openCL);
                out.write("\n");
                out.close();
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(1);
            }
            buildOpenCLContext(this.kernel.checkTaskType(), openCL,
                this.myOpenCLContextHandle);
            System.exit(0);
         }
         // at this point, i = the actual used number of arguments
         // (private buffers do not get treated as arguments)

         // Send the string to OpenCL to compile it
         buildOpenCLContext(this.kernel.checkTaskType(), openCL,
                 this.myOpenCLContextHandle);

         List<Long> dataHandlesForType =
             openclDataHandles.get(kernel.checkTaskType());
         synchronized (dataHandlesForType) {
             long dataHandle;
             if (dataHandlesForType.isEmpty()) {
                 dataHandle = initOpenCLData();
             } else {
                 dataHandle = dataHandlesForType.remove(0);
             }
             if (dataHandle == 0L) {
                 throw new RuntimeException("Error creating data handle");
             }
             this.myOpenCLDataHandle = dataHandle;
         }

         initJNIContextFromOpenCLDataContext(jniContextHandle,
             this.myOpenCLDataHandle);

         setArgsJNI(jniContextHandle,
             openclProgramContextHandles.get(kernel.checkTaskType()), args, argc);
      }
   }

   public synchronized Kernel execute(String _entrypointName,
           final Range _range, int deviceId, int deviceSlot, final int _passes, final boolean enableStrided,
           final boolean isRelaunch, final boolean dryRun, int taskId,
           int attemptId, String label) {

      long executeStartTime = System.currentTimeMillis();
      Kernel ret = kernel;

      if (_range == null) {
         throw new IllegalStateException("range can't be null");
      }

      // See if user supplied a Device
      Device device = _range.getDevice();

      if ((device != null) && (device instanceof OpenCLDevice)) {
            // Should be a no-op except when called by translate.sh script
            doEntrypointInit(_entrypointName, enableStrided, device, deviceId, deviceSlot,
                dryRun, taskId, attemptId);
            try {
               ret = executeOpenCL(_entrypointName, _range, _passes,
                       enableStrided, isRelaunch, label);
            } catch (final AparapiException e) {
               throw new RuntimeException(e);
            }
      } else {
         throw new RuntimeException("OpenCL was requested but Device supplied was not an OpenCLDevice");
      }

      if (Config.enableExecutionModeReporting) {
         System.out.println(kernel.getClass().getCanonicalName() + ":" + kernel.getExecutionMode());
      }

      executionTime = System.currentTimeMillis() - executeStartTime;
      accumulatedExecutionTime += executionTime;

      return ret;
   }


   private int getPrimitiveSize(int type) {
      if ((type & ARG_FLOAT) != 0) {
         return 4;
      } else if ((type & ARG_INT) != 0) {
         return 4;
      } else if ((type & ARG_BYTE) != 0) {
         return 1;
      } else if ((type & ARG_CHAR) != 0) {
         return 2;
      } else if ((type & ARG_BOOLEAN) != 0) {
         return 1;
      } else if ((type & ARG_SHORT) != 0) {
         return 2;
      } else if ((type & ARG_LONG) != 0) {
         return 8;
      } else if ((type & ARG_DOUBLE) != 0) {
         return 8;
      }
      return 0;
   }

   private final Set<Object> puts = new HashSet<Object>();

   /**
    * Enqueue a request to return this array from the GPU. This method blocks until the array is available.
    * <br/>
    * Note that <code>Kernel.put(type [])</code> calls will delegate to this call.
    * <br/>
    * Package public
    * 
    * @param array
    *          It is assumed that this parameter is indeed an array (of int, float, short etc).
    * 
    * @see Kernel#get(int[] arr)
    * @see Kernel#get(short[] arr)
    * @see Kernel#get(float[] arr)
    * @see Kernel#get(double[] arr)
    * @see Kernel#get(long[] arr)
    * @see Kernel#get(char[] arr)
    * @see Kernel#get(boolean[] arr)
    */
   public void get(Object array) {
      if (explicit
            && ((kernel.getExecutionMode() == Kernel.EXECUTION_MODE.GPU) || (kernel.getExecutionMode() == Kernel.EXECUTION_MODE.CPU))) {
         // Only makes sense when we are using OpenCL
         getJNI(jniContextHandle, array);
      }
   }

   /**
    * Tag this array so that it is explicitly enqueued before the kernel is executed. <br/>
    * Note that <code>Kernel.put(type [])</code> calls will delegate to this call. <br/>
    * Package public
    * 
    * @param array
    *          It is assumed that this parameter is indeed an array (of int, float, short etc).
    * @see Kernel#put(int[] arr)
    * @see Kernel#put(short[] arr)
    * @see Kernel#put(float[] arr)
    * @see Kernel#put(double[] arr)
    * @see Kernel#put(long[] arr)
    * @see Kernel#put(char[] arr)
    * @see Kernel#put(boolean[] arr)
    */

   public void put(Object array) {
      if (explicit
            && ((kernel.getExecutionMode() == Kernel.EXECUTION_MODE.GPU) || (kernel.getExecutionMode() == Kernel.EXECUTION_MODE.CPU))) {
         // Only makes sense when we are using OpenCL
         puts.add(array);
      }
   }

   private boolean explicit = false;

   public void setExplicit(boolean _explicit) {
      explicit = _explicit;
   }

   public boolean isExplicit() {
      return (explicit);
   }

   /**
    * Determine the time taken to convert bytecode to OpenCL for first Kernel.execute(range) call.
    * 
    * @return The time spent preparing the kernel for execution using GPU
    * 
    */
   public long getConversionTime() {
      return conversionTime;
   }

   /**
    * Determine the execution time of the previous Kernel.execute(range) call.
    * 
    * @return The time spent executing the kernel (ms)
    * 
    */
   public long getExecutionTime() {
      return executionTime;
   }

   /**
    * Determine the accumulated execution time of all previous Kernel.execute(range) calls.
    * 
    * @return The accumulated time spent executing this kernel (ms)
    * 
    */
   public long getAccumulatedExecutionTime() {
      return accumulatedExecutionTime;
   }

   private static class KernelAndArgLines {
      public final String kernel;
      public final List<String> argLines;

      public KernelAndArgLines(String kernel, List<String> argLines) {
          this.kernel = kernel;
          this.argLines = argLines;
      }
   }
}
