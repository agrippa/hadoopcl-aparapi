package com.amd.aparapi.internal.jni;

import com.amd.aparapi.internal.annotation.UsedByJNICode;

/**
 * This class is intended to be used as a 'proxy' or 'facade' object for Java code to interact with JNI
 */
public abstract class RangeJNI{
   @UsedByJNICode protected int globalSize = 1;

   @UsedByJNICode protected int localSize = 1;
}
