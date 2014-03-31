package com.amd.aparapi.internal.opencl;

import java.util.logging.Level;
import java.util.logging.Logger;

import com.amd.aparapi.internal.jni.OpenCLJNI;

/**
 * This class is intended to be a singleton which determines if OpenCL is available upon startup of Aparapi
 */
public class OpenCLLoader extends OpenCLJNI{

   private static boolean openCLAvailable = false;

   private static final OpenCLLoader instance = new OpenCLLoader();

   static {
     final String arch = System.getProperty("os.arch");
     String aparapiLibraryName = null;

     if (arch.equals("amd64") || arch.equals("x86_64")) {
        aparapiLibraryName = "aparapi_x86_64";
     } else if (arch.equals("x86") || arch.equals("i386")) {
        aparapiLibraryName = "aparapi_x86";
     }
     if (aparapiLibraryName != null) {
        try {
           Runtime.getRuntime().loadLibrary(aparapiLibraryName);
           openCLAvailable = true;
        } catch (final UnsatisfiedLinkError e) {
           System.err.println("Check your environment. Failed to load aparapi native library " + aparapiLibraryName
                 + " or possibly failed to locate opencl native library (opencl.dll/opencl.so)."
                 + " Ensure that both are in your PATH (windows) or in LD_LIBRARY_PATH (linux).");
           System.exit(1);
        }
     }
   }

   /**
    * Retrieve a singleton instance of OpenCLLoader
    * 
    * @return A singleton instance of OpenCLLoader
    */
   protected static OpenCLLoader getInstance() {
      return instance;
   }

   /**
    * Retrieve the status of whether OpenCL was successfully loaded
    * 
    * @return The status of whether OpenCL was successfully loaded
    */
   public static boolean isOpenCLAvailable() {
      return openCLAvailable;
   }
}
