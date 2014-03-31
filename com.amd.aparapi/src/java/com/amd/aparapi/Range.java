package com.amd.aparapi;

import java.util.Arrays;

import com.amd.aparapi.device.Device;
import com.amd.aparapi.internal.jni.RangeJNI;

/**
 * 
 * A representation of 1, 2 or 3 dimensional range of execution. 
 * 
 * This class uses factory methods to allow one, two or three dimensional ranges to be created. 
 * <br/>
 * For a Kernel operating over the linear range 0..1024 without a specified groups size we would create a one dimensional <code>Range</code> using 
 * <blockquote><pre>Range.create(1024);</pre></blockquote>
 * To request the same linear range but with a groupSize of 64 (range must be a multiple of group size!) we would use
 * <blockquote><pre>Range.create(1024,64);</pre></blockquote>
 * To request a two dimensional range over a grid (0..width)x(0..height) where width==512 and height=256 we would use
 * <blockquote><pre>
 * int width=512;
 * int height=256;
 * Range.create2D(width,height)
 * </pre></blockquote>
 * Again the above does not specify the group size.  One will be chosen for you. If you want to specify the groupSize (say 16x8; 16 wide by 8 high) use
 * <blockquote><pre>
 * int width=512;
 * int height=256;
 * int groupWidth=16;
 * int groupHeight=8;
 * Range.create2D(width, height, groupWidth, groupHeight);
 * </pre></blockquote>
 * Finally we can request a three dimensional range using 
 * <blockquote><pre>
 * int width=512;
 * int height=256;
 * int depth=8;
 * Range.create3D(width, height, depth);
 * </pre></blockquote>
 * And can specify a group size using 
 * <blockquote><pre>
 *  int width=512;
 *  int height=256;
 *  int depth=8;
 *  int groupWidth=8;
 *  int groupHeight=4;
 *  int groupDepth=2
 *  Range.create3D(width, height, depth, groupWidth, groupHeight, groupDepth);
 * </pre></blockquote>
 */
public class Range extends RangeJNI{

   public static final int THREADS_PER_CORE = 16;

   public static final int MAX_OPENCL_GROUP_SIZE = 256;

   public static final int MAX_GROUP_SIZE = Math.max(Runtime.getRuntime().availableProcessors() * THREADS_PER_CORE,
         MAX_OPENCL_GROUP_SIZE);

   private Device device = null;

   private int maxWorkGroupSize;

   private int[] maxWorkItemSize = new int[] {
         MAX_GROUP_SIZE,
         MAX_GROUP_SIZE,
         MAX_GROUP_SIZE
   };

   /**
    * Minimal constructor
    * 
    * @param _device
    * @param _dims
    */
   public Range(Device _device) {
      device = _device;

      if (device != null) {
         maxWorkItemSize = device.getMaxWorkItemSize();
         maxWorkGroupSize = device.getMaxWorkGroupSize();
      } else {
         maxWorkGroupSize = MAX_GROUP_SIZE;
      }
   }

   /** 
    * Create a one dimensional range <code>0.._globalWidth</code> which is processed in groups of size _localWidth.
    * <br/>
    * Note that for this range to be valid : </br> <strong><code>_globalWidth > 0 && _localWidth > 0 && _localWidth < MAX_GROUP_SIZE && _globalWidth % _localWidth==0</code></strong>
    * 
    * @param _globalWidth the overall range we wish to process
    * @param _localWidth the size of the group we wish to process.
    * @return A new Range with the requested dimensions
    */
   public static Range create(Device _device, int _globalWidth, int _localWidth) {
      final Range range = new Range(_device);

      range.setGlobalSize(_globalWidth);
      range.setLocalSize(_localWidth);

      return (range);
   }

   /**
    * Determine the set of factors for a given value.
    * @param _value The value we wish to factorize. 
    * @param _max an upper bound on the value that can be chosen
    * @return and array of factors of _value
    */

   private static int[] getFactors(int _value, int _max) {
      final int factors[] = new int[MAX_GROUP_SIZE];
      int factorIdx = 0;

      for (int possibleFactor = 1; possibleFactor <= _max; possibleFactor++) {
         if ((_value % possibleFactor) == 0) {
            factors[factorIdx++] = possibleFactor;
         }
      }

      return (Arrays.copyOf(factors, factorIdx));
   }

   /** 
    * Create a one dimensional range <code>0.._globalWidth</code> with an undefined group size.
    * <br/>
    * Note that for this range to be valid :- </br> <strong><code>_globalWidth > 0 </code></strong>
    * <br/>
    * The groupsize will be chosen such that _localWidth > 0 && _localWidth < MAX_GROUP_SIZE && _globalWidth % _localWidth==0 is true
    * 
    * We extract the factors of _globalWidth and choose the highest value.
    * 
    * @param _globalWidth the overall range we wish to process
    * @return A new Range with the requested dimensions
    */
   public static Range create(Device _device, int _globalWidth) {
      final Range withoutLocal = create(_device, _globalWidth, 1);

      final int[] factors = getFactors(withoutLocal.getGlobalSize(), withoutLocal.getMaxWorkItemSize()[0]);
      withoutLocal.setLocalSize(factors[factors.length - 1]);
      return (withoutLocal);
   }

   public static Range create(int _globalWidth, int _localWidth) {
      final Range range = create(null, _globalWidth, _localWidth);

      return (range);
   }

   public static Range create(int _globalWidth) {
      final Range range = create(null, _globalWidth);

      return (range);
   }

   /**
    * Override {@link #toString()}
    */
   @Override public String toString() {
      final StringBuilder sb = new StringBuilder();
      sb.append("global:" + globalSize + " local:" + localSize);
      return (sb.toString());
   }

   /**
    * Get the localSize (of the group) given the requested dimension
    * 
    * @param _dim 0=width, 1=height, 2=depth
    * @return The size of the group give the requested dimension
    */
   public int getLocalSize() {
      return localSize;
   }

   /**
    * Get the globalSize (of the range) given the requested dimension
    * 
    * @param _dim 0=width, 1=height, 2=depth
    * @return The size of the group give the requested dimension
    */
   public int getGlobalSize() {
      return globalSize;
   }

   /**
    * Get the number of groups for the given dimension. 
    * 
    * <p>
    * This will essentially return globalXXXX/localXXXX for the given dimension (width, height, depth)
    * @param _dim The dim we are interested in 0, 1 or 2
    * @return the number of groups for the given dimension. 
    */
   public int getNumGroups() {
      return globalSize / localSize;
   }

   /**
    * 
    * @return The product of all valid localSize dimensions
    */
   public int getWorkGroupSize() {
      return localSize;
   }

   public Device getDevice() {
      return (device);
   }

   /**
    * @param globalSize_0
    *          the globalSize_0 to set
    */
   public void setGlobalSize(int globalSize) {
      this.globalSize = globalSize;
   }

   /**
    * @param localSize_0
    *          the localSize_0 to set
    */
   public void setLocalSize(int localSize) {
      this.localSize = localSize;
   }

   /**
    * @return the maxWorkGroupSize
    */
   public int getMaxWorkGroupSize() {
      return maxWorkGroupSize;
   }

   /**
    * @param maxWorkGroupSize
    *          the maxWorkGroupSize to set
    */
   public void setMaxWorkGroupSize(int maxWorkGroupSize) {
      this.maxWorkGroupSize = maxWorkGroupSize;
   }

   /**
    * @return the maxWorkItemSize
    */
   public int[] getMaxWorkItemSize() {
      return maxWorkItemSize;
   }

   /**
    * @param maxWorkItemSize
    *          the maxWorkItemSize to set
    */
   public void setMaxWorkItemSize(int[] maxWorkItemSize) {
      this.maxWorkItemSize = maxWorkItemSize;
   }
}
