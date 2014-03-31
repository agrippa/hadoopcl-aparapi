package com.amd.aparapi.device;

import com.amd.aparapi.Range;
import com.amd.aparapi.device.OpenCLDevice.DeviceComparitor;
import com.amd.aparapi.device.OpenCLDevice.DeviceSelector;

public abstract class Device{

   public static enum TYPE {
      UNKNOWN ("UNKNOWN"),
      GPU ("GPU"),
      CPU ("CPU"),
      JTP ("JTP"),
      SEQ ("SEQ"),
      JAVA ("JAVA");

      private final String name;
      private TYPE(String n) {
        this.name = n;
      }
      public String toString() {
        return this.name;
      }
   }

   public static Device best() {
      return (OpenCLDevice.select(new DeviceComparitor(){
         @Override public OpenCLDevice select(OpenCLDevice _deviceLhs, OpenCLDevice _deviceRhs) {
            if (_deviceLhs.getType() != _deviceRhs.getType()) {
               if (_deviceLhs.getType() == TYPE.GPU) {
                  return (_deviceLhs);
               } else {
                  return (_deviceRhs);
               }
            }

            if (_deviceLhs.getMaxComputeUnits() > _deviceRhs.getMaxComputeUnits()) {
               return (_deviceLhs);
            } else {
               return (_deviceRhs);
            }
         }
      }));
   }

   public static Device first(final Device.TYPE _type) {
      return (OpenCLDevice.select(new DeviceSelector(){
         @Override public OpenCLDevice select(OpenCLDevice _device) {
            return (_device.getType() == _type ? _device : null);
         }
      }));
   }

   public static Device firstGPU() {
      return (first(Device.TYPE.GPU));
   }

   public static Device firstCPU() {
      return (first(Device.TYPE.CPU));

   }

   protected TYPE type = TYPE.UNKNOWN;

   protected int maxWorkGroupSize;

   protected int maxWorkItemDimensions;

   protected int[] maxWorkItemSize = new int[] {
         0,
         0,
         0
   };

   public TYPE getType() {
      return type;
   }

   public void setType(TYPE type) {
      this.type = type;
   }

   public int getMaxWorkItemDimensions() {
      return maxWorkItemDimensions;
   }

   public void setMaxWorkItemDimensions(int _maxWorkItemDimensions) {
      maxWorkItemDimensions = _maxWorkItemDimensions;
   }

   public int getMaxWorkGroupSize() {
      return maxWorkGroupSize;
   }

   public void setMaxWorkGroupSize(int _maxWorkGroupSize) {
      maxWorkGroupSize = _maxWorkGroupSize;
   }

   public int[] getMaxWorkItemSize() {
      return maxWorkItemSize;
   }

   public void setMaxWorkItemSize(int[] maxWorkItemSize) {
      this.maxWorkItemSize = maxWorkItemSize;
   }

   public Range createRange(int _globalWidth) {
      return (Range.create(this, _globalWidth));
   }

   public Range createRange(int _globalWidth, int _localWidth) {
      return (Range.create(this, _globalWidth, _localWidth));
   }
}
