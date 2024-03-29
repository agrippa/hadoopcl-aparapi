package com.amd.aparapi.internal.kernel;

import java.lang.reflect.Field;
import java.nio.ByteBuffer;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.internal.jni.KernelArgJNI;
import com.amd.aparapi.internal.model.ClassModel;

/**
 * Each field (or captured field in the case of an anonymous inner class) referenced by any bytecode reachable from the users Kernel.run(), will
 * need to be represented as a <code>KernelArg</code>.
 * 
 * @see com.amd.aparapi.Kernel#execute(int _globalSize)
 * 
 * @author gfrost
 * 
 */
public class KernelArg extends KernelArgJNI{

   /**
    * The byte array for obj conversion passed to opencl
    */
   private byte[] objArrayBuffer;

   /**
    * The ByteBuffer fronting the byte array
    */
   private ByteBuffer objArrayByteBuffer;

   /**
    * ClassModel of the array elements (not used on JNI side)
    * 
    */
   private ClassModel objArrayElementModel;

   /**
    * Only set for AparapiBuffer objs,
    */
   private Object primitiveBuf;

   /**
    * Size of this primitive
    */
   private int primitiveSize;

   /**
    * Default constructor
    */
   protected KernelArg() {

   }

   /**
    * @return the objArrayBuffer
    */
   protected byte[] getObjArrayBuffer() {
      return objArrayBuffer;
   }

   /**
    * @param objArrayBuffer the objArrayBuffer to set
    */
   protected void setObjArrayBuffer(byte[] objArrayBuffer) {
      this.objArrayBuffer = objArrayBuffer;
   }

   /**
    * @return the objArrayByteBuffer
    */
   protected ByteBuffer getObjArrayByteBuffer() {
      return objArrayByteBuffer;
   }

   /**
    * @param objArrayByteBuffer the objArrayByteBuffer to set
    */
   protected void setObjArrayByteBuffer(ByteBuffer objArrayByteBuffer) {
      this.objArrayByteBuffer = objArrayByteBuffer;
   }

   /**
    * @return the objArrayElementModel
    */
   protected ClassModel getObjArrayElementModel() {
      return objArrayElementModel;
   }

   /**
    * @param objArrayElementModel the objArrayElementModel to set
    */
   protected void setObjArrayElementModel(ClassModel objArrayElementModel) {
      this.objArrayElementModel = objArrayElementModel;
   }

   /**
    * @return the primitiveBuf
    */
   protected Object getPrimitiveBuf() {
      return primitiveBuf;
   }

   /**
    * @param primitiveBuf the primitiveBuf to set
    */
   protected void setPrimitiveBuf(Object primitiveBuf) {
      this.primitiveBuf = primitiveBuf;
   }

   /**
    * @return the primitiveSize
    */
   public int getPrimitiveSize() {
      return primitiveSize;
   }

   /**
    * @param primitiveSize the primitiveSize to set
    */
   public void setPrimitiveSize(int primitiveSize) {
      this.primitiveSize = primitiveSize;
   }

   /**
    * @return the type
    */
   public int getType() {
      return type;
   }

   /**
    * @param type the type to set
    */
   public void setType(int type) {
      this.type = type;
   }

   /**
    * @return the name
    */
   public String getName() {
      return name;
   }

   /**
    * @param name the name to set
    */
   public void setName(String name) {
      this.name = name;
   }

   /**
    * @return the javaArray
    */
   protected Object getJavaArray() {
      return javaArray;
   }

   /**
    * @param javaArray the javaArray to set
    */
   protected void setJavaArray(Object javaArray) {
      this.javaArray = javaArray;
   }

   /**
    * @return the sizeInBytes
    */
   public long getSizeInBytes() {
      return sizeInBytes;
   }

   /**
    * @param sizeInBytes the sizeInBytes to set
    */
   public void setSizeInBytes(long sizeInBytes) {
      this.sizeInBytes = sizeInBytes;
   }

   /**
    * @return the numElements
    */
   public int getNumElements() {
      return numElements;
   }

   /**
    * @param numElements the numElements to set
    */
   public void setNumElements(int numElements) {
      this.numElements = numElements;
   }

   /**
    * @return the array
    */
   // protected Object getArray() {
   //    return array;
   // }

   /**
    * @param array the array to set
    */
   // protected void setArray(Object array) {
   //    this.array = array;
   // }

   /**
    * @return the field
    */
   protected Field getField() {
      return field;
   }

   /**
    * @param field the field to set
    */
   protected void setField(Field field) {
      this.field = field;
   }

   /**
    * @return the buffer
    */
   // protected Object getJavaBuffer() {
   //    return javaBuffer;
   // }

   /**
    * @param buffer the buffer to set
    */
   // protected void setJavaBuffer(Object buffer) {
   //    this.javaBuffer = buffer;
   // }

   /**
    * @return the number of dimensions to buffer
    */
   public int getNumDims() {
      return numDims;
   }

   /**
    * @param numDims the number of dimensions for the buffer
    */
   public void setNumDims(int numDims) {
      this.numDims = numDims;
   }

   /**
    * @return the dimensions for the buffer
    */
   public int[] getDims() {
      return dims;
   }

   /**
    * @param dims the dimsensions for the buffer
    */
   public void setDims(int[] dims) {
      this.dims = dims;
   }
}
