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
package com.amd.aparapi.internal.writer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.LinkedList;
import java.util.Stack;
import java.util.Set;
import java.util.HashSet;
import java.util.Arrays;

import com.amd.aparapi.Config;
import com.amd.aparapi.Kernel;
import com.amd.aparapi.internal.exception.CodeGenException;
import com.amd.aparapi.internal.instruction.Instruction;
import com.amd.aparapi.internal.instruction.InstructionSet;
import com.amd.aparapi.internal.instruction.InstructionSet.AccessLocalVariable;
import com.amd.aparapi.internal.instruction.InstructionSet.AccessArrayElement;
import com.amd.aparapi.internal.instruction.InstructionSet.AccessField;
import com.amd.aparapi.internal.instruction.InstructionSet.AssignToArrayElement;
import com.amd.aparapi.internal.instruction.InstructionSet.AssignToField;
import com.amd.aparapi.internal.instruction.InstructionSet.AssignToLocalVariable;
import com.amd.aparapi.internal.instruction.InstructionSet.BinaryOperator;
import com.amd.aparapi.internal.instruction.InstructionSet.I_ALOAD_0;
import com.amd.aparapi.internal.instruction.InstructionSet.I_GETFIELD;
import com.amd.aparapi.internal.instruction.InstructionSet.I_INVOKESPECIAL;
import com.amd.aparapi.internal.instruction.InstructionSet.I_IUSHR;
import com.amd.aparapi.internal.instruction.InstructionSet.I_LUSHR;
import com.amd.aparapi.internal.instruction.InstructionSet.MethodCall;
import com.amd.aparapi.internal.instruction.InstructionSet.VirtualMethodCall;
import com.amd.aparapi.internal.model.ClassModel;
import com.amd.aparapi.internal.model.ClassModel.ClassModelField;
import com.amd.aparapi.internal.model.ClassModel.LocalVariableInfo;
import com.amd.aparapi.internal.model.ClassModel.LocalVariableTableEntry;
import com.amd.aparapi.internal.model.ClassModel.AttributePool.RuntimeAnnotationsEntry;
import com.amd.aparapi.internal.model.ClassModel.AttributePool.RuntimeAnnotationsEntry.AnnotationInfo;
import com.amd.aparapi.internal.model.ClassModel.ConstantPool.FieldEntry;
import com.amd.aparapi.internal.model.ClassModel.ConstantPool.MethodEntry;
import com.amd.aparapi.internal.model.Entrypoint;
import com.amd.aparapi.internal.model.MethodModel;
import com.amd.aparapi.opencl.OpenCL.Constant;
import com.amd.aparapi.opencl.OpenCL.Local;

public abstract class KernelWriter extends BlockWriter{
   public static HadoopTypes types = null;
   protected abstract String removePreviousLine();

   private final String cvtBooleanToChar = "char ";

   private final String cvtBooleanArrayToCharStar = "char* ";

   private final String cvtByteToChar = "char ";

   private final String cvtByteArrayToCharStar = "char* ";

   private final String cvtCharToShort = "unsigned short ";

   private final String cvtCharArrayToShortStar = "unsigned short* ";

   private final String cvtIntArrayToIntStar = "int* ";

   private final String cvtFloatArrayToFloatStar = "float* ";

   private final String cvtDoubleArrayToDoubleStar = "double* ";

   private final String cvtLongArrayToLongStar = "long* ";

   private final String cvtShortArrayToShortStar = "short* ";

   // private static Logger logger = Logger.getLogger(Config.getLoggerName());

   private Entrypoint entryPoint = null;


   public final static Map<String, String> javaToCLIdentifierMap = new HashMap<String, String>();
   {
      javaToCLIdentifierMap.put("getGlobalId()I", "get_global_id(0)");
      javaToCLIdentifierMap.put("getGlobalId(I)I", "get_global_id"); // no parenthesis if we are conveying args
      javaToCLIdentifierMap.put("getGlobalX()I", "get_global_id(0)");
      javaToCLIdentifierMap.put("getGlobalY()I", "get_global_id(1)");
      javaToCLIdentifierMap.put("getGlobalZ()I", "get_global_id(2)");

      javaToCLIdentifierMap.put("getGlobalSize()I", "get_global_size(0)");
      javaToCLIdentifierMap.put("getGlobalSize(I)I", "get_global_size"); // no parenthesis if we are conveying args
      javaToCLIdentifierMap.put("getGlobalWidth()I", "get_global_size(0)");
      javaToCLIdentifierMap.put("getGlobalHeight()I", "get_global_size(1)");
      javaToCLIdentifierMap.put("getGlobalDepth()I", "get_global_size(2)");

      javaToCLIdentifierMap.put("getLocalId()I", "get_local_id(0)");
      javaToCLIdentifierMap.put("getLocalId(I)I", "get_local_id"); // no parenthesis if we are conveying args
      javaToCLIdentifierMap.put("getLocalX()I", "get_local_id(0)");
      javaToCLIdentifierMap.put("getLocalY()I", "get_local_id(1)");
      javaToCLIdentifierMap.put("getLocalZ()I", "get_local_id(2)");

      javaToCLIdentifierMap.put("getLocalSize()I", "get_local_size(0)");
      javaToCLIdentifierMap.put("getLocalSize(I)I", "get_local_size"); // no parenthesis if we are conveying args
      javaToCLIdentifierMap.put("getLocalWidth()I", "get_local_size(0)");
      javaToCLIdentifierMap.put("getLocalHeight()I", "get_local_size(1)");
      javaToCLIdentifierMap.put("getLocalDepth()I", "get_local_size(2)");

      javaToCLIdentifierMap.put("getNumGroups()I", "get_num_groups(0)");
      javaToCLIdentifierMap.put("getNumGroups(I)I", "get_num_groups"); // no parenthesis if we are conveying args
      javaToCLIdentifierMap.put("getNumGroupsX()I", "get_num_groups(0)");
      javaToCLIdentifierMap.put("getNumGroupsY()I", "get_num_groups(1)");
      javaToCLIdentifierMap.put("getNumGroupsZ()I", "get_num_groups(2)");

      javaToCLIdentifierMap.put("getGroupId()I", "get_group_id(0)");
      javaToCLIdentifierMap.put("getGroupId(I)I", "get_group_id"); // no parenthesis if we are conveying args
      javaToCLIdentifierMap.put("getGroupX()I", "get_group_id(0)");
      javaToCLIdentifierMap.put("getGroupY()I", "get_group_id(1)");
      javaToCLIdentifierMap.put("getGroupZ()I", "get_group_id(2)");

      javaToCLIdentifierMap.put("getPassId()I", "get_pass_id(this)");

      javaToCLIdentifierMap.put("localBarrier()V", "barrier(CLK_LOCAL_MEM_FENCE)");

      javaToCLIdentifierMap.put("globalBarrier()V", "barrier(CLK_GLOBAL_MEM_FENCE)");
   }

   /**
    * These three convert functions are here to perform
    * any type conversion that may be required between
    * Java and OpenCL.
    * 
    * @param _typeDesc
    *          String in the Java JNI notation, [I, etc
    * @return Suitably converted string, "char*", etc
    */
   @Override public String convertType(String _typeDesc, boolean useClassModel) {
      if (_typeDesc.equals("Z") || _typeDesc.equals("boolean")) {
         return (cvtBooleanToChar);
      } else if (_typeDesc.equals("[Z") || _typeDesc.equals("boolean[]")) {
         return (cvtBooleanArrayToCharStar);
      } else if (_typeDesc.equals("B") || _typeDesc.equals("byte")) {
         return (cvtByteToChar);
      } else if (_typeDesc.equals("[B") || _typeDesc.equals("byte[]")) {
         return (cvtByteArrayToCharStar);
      } else if (_typeDesc.equals("C") || _typeDesc.equals("char")) {
         return (cvtCharToShort);
      } else if (_typeDesc.equals("[C") || _typeDesc.equals("char[]")) {
         return (cvtCharArrayToShortStar);
      } else if (_typeDesc.equals("[I") || _typeDesc.equals("int[]")) {
         return (cvtIntArrayToIntStar);
      } else if (_typeDesc.equals("[F") || _typeDesc.equals("float[]")) {
         return (cvtFloatArrayToFloatStar);
      } else if (_typeDesc.equals("[D") || _typeDesc.equals("double[]")) {
         return (cvtDoubleArrayToDoubleStar);
      } else if (_typeDesc.equals("[J") || _typeDesc.equals("long[]")) {
         return (cvtLongArrayToLongStar);
      } else if (_typeDesc.equals("[S") || _typeDesc.equals("short[]")) {
         return (cvtShortArrayToShortStar);
      }
      // if we get this far, we haven't matched anything yet
      if (useClassModel) {
         return (ClassModel.convert(_typeDesc, "", true));
      } else {
         return _typeDesc;
      }
   }

   @Override public void writeMethod(MethodCall _methodCall, MethodEntry _methodEntry) throws CodeGenException {

      // System.out.println("_methodEntry = " + _methodEntry);
      // special case for buffers
      Set<String> modifiableMethods = new HashSet<String>(
             Arrays.asList("next", "seekTo", "current", 
                 "getVal1", "getVal2", "getValId", "getValIndices",
                 "getValVals", "vectorLength", "currentVectorLength",
                 "get"));
      final int argc = _methodEntry.getStackConsumeCount();

      final String methodName = _methodEntry.getNameAndTypeEntry().getNameUTF8Entry().getUTF8();
      final String methodSignature = _methodEntry.getNameAndTypeEntry().getDescriptorUTF8Entry().getUTF8();

      final String barrierAndGetterMappings = javaToCLIdentifierMap.get(methodName + methodSignature);

      if (barrierAndGetterMappings != null) {
         // this is one of the OpenCL barrier or size getter methods
         // write the mapping and exit
         if (argc > 0) {
            write(barrierAndGetterMappings);
            write("(");
            for (int arg = 0; arg < argc; arg++) {
               if ((arg != 0)) {
                  write(", ");
               }
               writeInstruction(_methodCall.getArg(arg));
            }
            write(")");
         } else {
            write(barrierAndGetterMappings);
         }
      } else {

         final String intrinsicMapping = Kernel.getMappedMethodName(_methodEntry);
         // System.out.println("getMappedMethodName for " + methodName + " returned " + mapping);
         boolean isIntrinsic = false;

         if (intrinsicMapping == null) {
            assert entryPoint != null : "entryPoint should not be null";
            final boolean isSpecial = _methodCall instanceof I_INVOKESPECIAL;
            boolean isMapped = Kernel.isMappedMethod(_methodEntry);
            final MethodModel m = entryPoint.getCallTarget(_methodEntry, isSpecial);

            if (m != null) {
               write(m.getName());
            } else {
               // Must be a library call like rsqrt
               assert isMapped : _methodEntry + " should be mapped method!";
               write(methodName);
               isIntrinsic = true;
            }
         } else {
            write(intrinsicMapping);
         }

         write("(");

         if ((intrinsicMapping == null) && (_methodCall instanceof VirtualMethodCall) && (!isIntrinsic)) {

            final Instruction i = ((VirtualMethodCall) _methodCall).getInstanceReference();

            if (i instanceof I_ALOAD_0) {
               write("this");
            } else if (i instanceof AccessArrayElement) {
               final AccessArrayElement arrayAccess = (AccessArrayElement) ((VirtualMethodCall) _methodCall).getInstanceReference();
               final Instruction refAccess = arrayAccess.getArrayRef();
               //assert refAccess instanceof I_GETFIELD : "ref should come from getfield";
               final String fieldName = ((AccessField) refAccess).getConstantPoolFieldEntry().getNameAndTypeEntry()
                     .getNameUTF8Entry().getUTF8();
               write(" &(this->" + fieldName);
               write("[");
               writeInstruction(arrayAccess.getArrayIndex());
               write("])");
            } else {
               assert false : "unhandled call from: " + i;
            }
         }
         if( modifiableMethods.contains(methodName)) {
             write("this");
         }

         for (int arg = 0; arg < argc; arg++) {
            if (((intrinsicMapping == null) && (_methodCall instanceof VirtualMethodCall) && (!isIntrinsic)) || (arg != 0) || (modifiableMethods.contains(methodName))) {
               write(", ");
            }
            Instruction argObj = _methodCall.getArg(arg);
            if (argObj instanceof AccessLocalVariable) {
                final AccessLocalVariable localVariableLoadInstruction = (AccessLocalVariable) argObj;
                final LocalVariableInfo localVariable = localVariableLoadInstruction.getLocalVariableInfo();
                if(localVariable.isArray()) {
                    final boolean isSpecial = _methodCall instanceof I_INVOKESPECIAL;
                    final MethodModel m =
                        entryPoint.getCallTarget(_methodEntry, isSpecial);
                    LocalVar beingPassed = new LocalVar(this.currentMethodBody,
                            localVariable.getVariableName());
                    LocalVar argument = new LocalVar(m == null ? methodName : m.getName(),
                            arg);
                    VarAlias newAlias = new VarAlias(beingPassed, argument);
                    this.addAlias(newAlias);
                }
                /*
                System.out.println("      \""+localVariable.getVariableName()+"\" "+argObj.getClass().toString());
                */
            } else {
                /*
                System.out.println("      \""+argObj.toString()+"\" "+argObj.getClass().toString());
                */
            }
            writeInstruction(argObj);
         }
         write(")");
      }
   }

   public void writePragma(String _name, boolean _enable) {
      write("#pragma OPENCL EXTENSION " + _name + " : " + (_enable ? "en" : "dis") + "able");
      newLine();
   }

   public final static String __local = "__local";

   public final static String __global = "__global";

   public final static String __constant = "__constant";

   public final static String LOCAL_ANNOTATION_NAME = "L" + Local.class.getName().replace(".", "/") + ";";

   public final static String CONSTANT_ANNOTATION_NAME = "L" + Constant.class.getName().replace(".", "/") + ";";

   private void hadoopOutputWrite(String argToken) {
         String[] tokens = argToken.split(" ");
         String varname = tokens[2];
         char[] chars = varname.toCharArray();
         chars[0] = Character.toUpperCase(chars[0]);
         String capitalizedVarname = null;
         if(Character.isDigit(chars[chars.length-1])) {
             char save = chars[chars.length-1];
             chars[chars.length-1] = 's';
             capitalizedVarname = String.valueOf(chars);
             capitalizedVarname += save;
         } else {
             capitalizedVarname = String.valueOf(chars);
             capitalizedVarname += "s";
         }

         write("   this->output"+capitalizedVarname+"[index] = "+varname+";\n");
   }

   private String[] getInputOutputTypes(String classNameStr) {
     String typeStr = classNameStr.substring(classNameStr.lastIndexOf(".") + 1);
     int[] typeIndices = { -1, -1, -1, -1, -1 };
     int typeCount = 0;
     int typeStrIndex = 0;
     while(typeCount < typeIndices.length) {
         if(Character.isUpperCase(typeStr.charAt(typeStrIndex))) {
             typeIndices[typeCount] = typeStrIndex;
             if(typeStr.charAt(typeStrIndex) == 'U') typeStrIndex++;
             typeCount++;
         }
         typeStrIndex++;
     }
     String inputKeyType = typeStr.substring(typeIndices[0], typeIndices[1]).toLowerCase();
     String inputValType = typeStr.substring(typeIndices[1], typeIndices[2]).toLowerCase();
     String outputKeyType = typeStr.substring(typeIndices[2], typeIndices[3]).toLowerCase();
     String outputValType = typeStr.substring(typeIndices[3], typeIndices[4]).toLowerCase();
     return new String[] { inputKeyType, inputValType, outputKeyType, outputValType };
   }

   @Override public void write(Entrypoint _entryPoint) throws CodeGenException {
      final List<String> thisStruct = new ArrayList<String>();
      final List<String> argLines = new ArrayList<String>();
      final List<String> assigns = new ArrayList<String>();

      entryPoint = _entryPoint;

      String enclosingClassName = entryPoint.getClassModel().getClassWeAreModelling().getName();
      String enclosingSuperClassName = entryPoint.getClassModel().getSuperClazz().getClassWeAreModelling().getName();
      boolean isMapper = enclosingSuperClassName.indexOf("Mapper") != -1;

      String inputKeyType = null;
      String inputValType = null;
      String outputKeyType = null;
      String outputValType = null;

      String[] types = getInputOutputTypes(enclosingSuperClassName);
      inputKeyType = types[0];
      inputValType = types[1];
      outputKeyType = types[2];
      outputValType = types[3];

      KernelWriter.types = new HadoopTypes(isMapper ? HADOOPTYPE.MAPPER : HADOOPTYPE.REDUCER, 
              inputKeyType, inputValType, outputKeyType, outputValType);

      //System.out.println("class=\""+enclosingClassName+"\"");
      //System.out.println("super class=\""+enclosingSuperClassName+"\"");

      for (final ClassModelField field : _entryPoint.getReferencedClassModelFields()) {
         // Field field = _entryPoint.getClassModel().getField(f.getName());
         final StringBuilder thisStructLine = new StringBuilder();
         final StringBuilder argLine = new StringBuilder();
         final StringBuilder assignLine = new StringBuilder();

         String signature = field.getDescriptor();

         boolean isPointer = false;

         int numDimensions = 0;

         // check the suffix 
         String type = field.getName().endsWith(Kernel.LOCAL_SUFFIX) ? __local
               : (field.getName().endsWith(Kernel.CONSTANT_SUFFIX) ? __constant : __global);
         final RuntimeAnnotationsEntry visibleAnnotations = field.getAttributePool().getRuntimeVisibleAnnotationsEntry();

         if (visibleAnnotations != null) {
            for (final AnnotationInfo ai : visibleAnnotations) {
               final String typeDescriptor = ai.getTypeDescriptor();
               if (typeDescriptor.equals(LOCAL_ANNOTATION_NAME)) {
                  type = __local;
               } else if (typeDescriptor.equals(CONSTANT_ANNOTATION_NAME)) {
                  type = __constant;
               }
            }
         }

         //if we have a an array we want to mark the object as a pointer
         //if we have a multiple dimensional array we want to remember the number of dimensions
         while (signature.startsWith("[")) {
            if(isPointer == false) {
               argLine.append(type + " ");
               thisStructLine.append(type + " ");
            }
            isPointer = true;
            numDimensions++;
            signature = signature.substring(1);
         }

         // If it is a converted array of objects, emit the struct param
         String className = null;
         if (signature.startsWith("L")) {
            // Turn Lcom/amd/javalabs/opencl/demo/DummyOOA; into com_amd_javalabs_opencl_demo_DummyOOA for example
            className = (signature.substring(1, signature.length() - 1)).replace("/", "_");
            // if (logger.isLoggable(Level.FINE)) {
            // logger.fine("Examining object parameter: " + signature + " new: " + className);
            // }

            argLine.append(className);
            thisStructLine.append(className);
         } else {
            argLine.append(convertType(ClassModel.typeName(signature.charAt(0)), false));
            thisStructLine.append(convertType(ClassModel.typeName(signature.charAt(0)), false));
         }

         argLine.append(" ");
         thisStructLine.append(" ");

         if (isPointer) {
            argLine.append("*");
            thisStructLine.append("*");
         }
         assignLine.append("this->");
         assignLine.append(field.getName());
         assignLine.append(" = ");
         assignLine.append(field.getName());
         argLine.append(field.getName());
         thisStructLine.append(field.getName());
         assigns.add(assignLine.toString());
         argLines.add(argLine.toString());
         thisStruct.add(thisStructLine.toString());

         // Add int field into "this" struct for supporting java arraylength op
         // named like foo__javaArrayLength
         if (isPointer && _entryPoint.getArrayFieldArrayLengthUsed().contains(field.getName()) ||
             isPointer && numDimensions > 1) {
            
            for(int i = 0; i < numDimensions; i++) {
               final StringBuilder lenStructLine = new StringBuilder();
               final StringBuilder lenArgLine = new StringBuilder();
               final StringBuilder lenAssignLine = new StringBuilder();
               final StringBuilder dimStructLine = new StringBuilder();
               final StringBuilder dimArgLine = new StringBuilder();
               final StringBuilder dimAssignLine = new StringBuilder();

               String lenName = field.getName() + BlockWriter.arrayLengthMangleSuffix +
                    Integer.toString(i);

               lenStructLine.append("int " + lenName);

               lenAssignLine.append("this->");
               lenAssignLine.append(lenName);
               lenAssignLine.append(" = ");
               lenAssignLine.append(lenName);

               lenArgLine.append("int " + lenName);

               assigns.add(lenAssignLine.toString());
               argLines.add(lenArgLine.toString());
               thisStruct.add(lenStructLine.toString());

               //String dimName = field.getName() + BlockWriter.arrayDimMangleSuffix +
               //     Integer.toString(i);

               //dimStructLine.append("int " + dimName);

               //dimAssignLine.append("this->");
               //dimAssignLine.append(dimName);
               //dimAssignLine.append(" = ");
               //dimAssignLine.append(dimName);

               //dimArgLine.append("int " + dimName);

               //assigns.add(dimAssignLine.toString());
               //argLines.add(dimArgLine.toString());
               //thisStruct.add(dimStructLine.toString());
            }
         }
      }

      if (Config.enableByteWrites || _entryPoint.requiresByteAddressableStorePragma()) {
         // Starting with OpenCL 1.1 (which is as far back as we support)
         // this feature is part of the core, so we no longer need this pragma
         if (false) {
            writePragma("cl_khr_byte_addressable_store", true);
            newLine();
         }
      }

      boolean usesAtomics = false;
      if (Config.enableAtomic32 || _entryPoint.requiresAtomic32Pragma()) {
         usesAtomics = true;
         writePragma("cl_khr_global_int32_base_atomics", true);
         writePragma("cl_khr_global_int32_extended_atomics", true);
         writePragma("cl_khr_local_int32_base_atomics", true);
         writePragma("cl_khr_local_int32_extended_atomics", true);
      }

      if (Config.enableAtomic64 || _entryPoint.requiresAtomic64Pragma()) {
         usesAtomics = true;
         writePragma("cl_khr_int64_base_atomics", true);
         writePragma("cl_khr_int64_extended_atomics", true);
      }

      if (usesAtomics) {
         write("int atomicAdd(__global int *_arr, int _index, int _delta){");
         in();
         {
            newLine();
            write("return atomic_add(&_arr[_index], _delta);");
            out();
            newLine();
         }
         write("}");

         newLine();
      }

      if (true /*Config.enableDoubles || _entryPoint.requiresDoublePragma()*/) {
         writePragma("cl_khr_fp64", true);
         newLine();
      }

      // Emit structs for oop transformation accessors
      for (final ClassModel cm : _entryPoint.getObjectArrayFieldsClasses().values()) {
         final ArrayList<FieldEntry> fieldSet = cm.getStructMembers();
         if (fieldSet.size() > 0) {
            final String mangledClassName = cm.getClassWeAreModelling().getName().replace(".", "_");
            newLine();
            write("typedef struct " + mangledClassName + "_s{");
            in();
            newLine();

            int totalSize = 0;
            int alignTo = 0;

            final Iterator<FieldEntry> it = fieldSet.iterator();
            while (it.hasNext()) {
               final FieldEntry field = it.next();
               final String fType = field.getNameAndTypeEntry().getDescriptorUTF8Entry().getUTF8();
               final int fSize = InstructionSet.TypeSpec.valueOf(fType.equals("Z") ? "B" : fType).getSize();

               if (fSize > alignTo) {
                  alignTo = fSize;
               }
               totalSize += fSize;

               final String cType = convertType(field.getNameAndTypeEntry().getDescriptorUTF8Entry().getUTF8(), true);
               assert cType != null : "could not find type for " + field.getNameAndTypeEntry().getDescriptorUTF8Entry().getUTF8();
               writeln(cType + " " + field.getNameAndTypeEntry().getNameUTF8Entry().getUTF8() + ";");
            }

            // compute total size for OpenCL buffer
            int totalStructSize = 0;
            if ((totalSize % alignTo) == 0) {
               totalStructSize = totalSize;
            } else {
               // Pad up if necessary
               totalStructSize = ((totalSize / alignTo) + 1) * alignTo;
            }
            if (totalStructSize > alignTo) {
               while (totalSize < totalStructSize) {
                  // structBuffer.put((byte)-1);
                  writeln("char _pad_" + totalSize + ";");
                  totalSize++;
               }
            }

            out();
            newLine();
            write("} " + mangledClassName + ";");
            newLine();
         }
      }

      write("typedef struct This_s{");

      in();
      newLine();
      for (final String line : thisStruct) {
         write(line);
         writeln(";");
      }
      //write("__local double *localGlobals;\n");
      //write("   __local int *localGlobalIndices;\n");
      //write("   unsigned long localMemSize;\n");
      if (!isMapper) {
          write("   int currentValueIndex;\n");
          write("   int currentStartingValueIndex;\n");
          write("   int currentNumValues;\n");
          if(inputValType.equals("svec")) {
              write("   int currentNumAuxValues;\n");
          }
      }
      write("   int reservedOffset;\n");
      write("   int reservedAuxOffset;\n");
      write("   int iter;\n");
      write("   int passid");
      out();
      writeln(";");
      // out();
      // newLine();
      write("}This;");
      newLine();
      newLine();
      write("#define NULL (0x0)");
      newLine();
      newLine();
      write("int get_pass_id(This *this){");
      in();
      {
         newLine();
         write("return this->passid;");
         out();
         newLine();
      }
      write("}");
      newLine();
      if(!isMapper) {
          write("int next(This *this) {\n");
          write("   if (this->currentValueIndex == this->currentStartingValueIndex+this->currentNumValues-1) return 0;\n");
          write("   this->currentValueIndex = this->currentValueIndex + 1;\n");
          write("   return 1;\n");
          write("}\n");
          write("\n");
          write("int seekTo(This *this, int set) {\n");
          write("   if (set < 0 || this->currentStartingValueIndex+set >= this->currentNumValues) return 0;\n");
          write("   this->currentValueIndex = this->currentStartingValueIndex+set;\n");
          write("   return 1;\n");
          write("}\n");
          write("\n");
          write("int current(This *this) {\n");
          write("   return this->currentValueIndex - this->currentStartingValueIndex;\n");
          write("}\n");
          write("\n");
          write("int nValues(This *this) {\n");
          write("   return this->currentNumValues;\n");
          write("}\n");
          write("\n");
      }

      if(!isMapper) {
          if(inputValType.equals("pair")) {
              write("double getVal1(This *this) {\n");
              write("   return this->inputVals1[this->currentValueIndex];\n");
              write("}\n");
              write("\n");
              write("double getVal2(This *this) {\n");
              write("   return this->inputVals2[this->currentValueIndex];\n");
              write("}\n");
              write("\n");
          } else if (inputValType.equals("upair")) {
              write("int getValId(This *this) {\n");
              write("   return this->inputValIds[this->currentValueIndex];\n");
              write("}\n");
              write("\n");
              write("double getVal1(This *this) {\n");
              write("   return this->inputVals1[this->currentValueIndex];\n");
              write("}\n");
              write("\n");
              write("double getVal2() {\n");
              write("   return this->inputVals2[this->currentValueIndex];\n");
              write("}\n");
              write("\n");
          } else if (inputValType.equals("svec")) {
              write("__global int *getValIndices(This *this) {\n");
              //write("   int end = (this->currentValueIndex == this->currentNumValues-1 ? this->currentNumAuxValues : this->inputValLookAsideBuffer[this->currentValueIndex+1]);
              write("   return this->inputValIndices + this->inputValLookAsideBuffer[this->currentValueIndex];\n");
              write("}\n");
              write("\n");
              write("__global double *getValVals(This *this) {\n");
              write("   return this->inputValVals + this->inputValLookAsideBuffer[this->currentValueIndex];\n");
              write("}\n");
              write("\n");
              write("int vectorLength(This *this, int index) {\n");
              write("   int start = this->inputValLookAsideBuffer[index];\n");
              write("   int end = (index == this->currentNumValues-1 ? this->currentNumAuxValues : this->inputValLookAsideBuffer[index+1]);\n");
              write("   return end-start;\n");
              write("}\n");
              write("\n");
              write("int currentVectorLength(This *this) {\n");
              write("   return vectorLength(this, this->currentValueIndex);\n");
              write("}\n");
              write("\n");
          } else {
              write(inputValType+" get(This *this) {\n");
              write("   return this->inputVals[this->currentValueIndex];\n");
              write("}\n");
              write("\n");
          }
      }

      final String mapreducePrefix = "org_apache_hadoop_mapreduce_";
      final String mapperWritePost = "HadoopCLMapperKernel__write";
      final String reducerWritePost = "HadoopCLReducerKernel__write";
      final String mapperRunPost = "HadoopCLMapperKernel__run";
      final String reducerRunPost = "HadoopCLReducerKernel__run";
      final String mapperCallPost = "HadoopCLMapperKernel__callMap";
      final String reducerCallPost = "HadoopCLReducerKernel__callReduce";
      final String reserveOutputPost = "__reserveOutput";
      final String accessOutputPost = "__accessOutput";
      final String getGlobalIndicesPost = "__getGlobalIndices";
      final String getGlobalValsPost = "__getGlobalVals";
      final String inputVectorLengthPost = "__inputVectorLength";
      final String allocIntPost = "__allocInt";
      final String allocDoublePost = "__allocDouble";
      final String mapPost = "__map";
      final String reducePost = "__reduce";

      HADOOPTYPE hadoopType = HADOOPTYPE.UNKNOWN;

      for (final MethodModel mm : _entryPoint.getCalledMethods()) {
         // write declaration :)

         boolean isMapWrite = false;
         boolean isReduceWrite = false;
         boolean isCallMap = false;
         boolean isCallReduce = false;
         boolean isReserveOutput = false;
         boolean isAccessOutput = false;
         boolean isGetGlobalIndices = false;
         boolean isGetGlobalVals = false;
         boolean isInputVectorLength = false;
         boolean isAllocInt = false;
         boolean isAllocDouble = false;
         boolean isReduce = false;
         boolean isMap = false;

         if(mm.getName().indexOf(mapreducePrefix) == 0) {
             if(mm.getName().indexOf(mapperWritePost) != -1) {
                 isMapWrite = true;
                 hadoopType = HADOOPTYPE.MAPPER;
             } else if(mm.getName().indexOf(reducerWritePost) != -1) {
                 isReduceWrite = true;
                 hadoopType = HADOOPTYPE.REDUCER;
             } else if(mm.getName().indexOf(mapperCallPost) != -1) {
                 isCallMap = true;
                 hadoopType = HADOOPTYPE.MAPPER;
             } else if(mm.getName().indexOf(reducerCallPost) != -1) {
                 isCallReduce = true;
                 hadoopType = HADOOPTYPE.REDUCER;
             } else if(mm.getName().indexOf(reserveOutputPost) != -1) {
                 isReserveOutput = true;
             } else if(mm.getName().indexOf(accessOutputPost) != -1) {
                 isAccessOutput = true;
             } else if(mm.getName().indexOf(getGlobalIndicesPost) != -1) {
                 isGetGlobalIndices = true;
             } else if(mm.getName().indexOf(getGlobalValsPost) != -1) {
                 isGetGlobalVals = true;
             } else if(mm.getName().indexOf(inputVectorLengthPost) != -1) {
                 isInputVectorLength = true;
             } else if(mm.getName().indexOf(allocIntPost) != -1) {
                 isAllocInt = true;
             } else if(mm.getName().indexOf(allocDoublePost) != -1) {
                 isAllocDouble = true;
             }
         }
         if(mm.getName().indexOf(mapPost) != -1) {
             isMap = true;
         } else if(mm.getName().indexOf(reducePost) != -1) {
             isReduce = true;
         }

         final String returnType = mm.getReturnType();
         // Arrays always map to __global arrays
         if (returnType.startsWith("[")) {
            write(" __global ");
         }
         write(convertType(returnType, true));

         write(mm.getName() + "(");

         if (!mm.getMethod().isStatic()) {
            if ((mm.getMethod().getClassModel() == _entryPoint.getClassModel())
                  || mm.getMethod().getClassModel().isSuperClass(_entryPoint.getClassModel().getClassWeAreModelling())) {
               write("This *this");
            } else {
               // Call to an object member or superclass of member
               for (final ClassModel c : _entryPoint.getObjectArrayFieldsClasses().values()) {
                  if (mm.getMethod().getClassModel() == c) {
                     write("__global " + mm.getMethod().getClassModel().getClassWeAreModelling().getName().replace(".", "_")
                           + " *this");
                     break;
                  } else if (mm.getMethod().getClassModel().isSuperClass(c.getClassWeAreModelling())) {
                     write("__global " + c.getClassWeAreModelling().getName().replace(".", "_") + " *this");
                     break;
                  }
               }
            }
         }

         boolean alreadyHasFirstArg = !mm.getMethod().isStatic();

         final LocalVariableTableEntry<LocalVariableInfo> lvte = mm.getLocalVariableTableEntry();
         List<String> gatherArgumentNames = new ArrayList<String>();
         for (final LocalVariableInfo lvi : lvte) {
            if ((lvi.getStart() == 0) && ((lvi.getVariableIndex() != 0) || mm.getMethod().isStatic())) { // full scope but skip this
               final String descriptor = lvi.getVariableDescriptor();
               if (alreadyHasFirstArg) {
                  write(", ");
               }

               // Arrays always map to __global arrays
               if (descriptor.startsWith("[")) {
                  write(" __global ");
               }

               write(convertType(descriptor, true));
               write(lvi.getVariableName());
               gatherArgumentNames.add(lvi.getVariableName());
               alreadyHasFirstArg = true;
            }
         }
         write(")");

         this.addMethodArgs(mm.getName(),
                 new MethodArgumentList(mm.getName(), gatherArgumentNames));

         if(isReduceWrite || isMapWrite) {
             try {
                 writeMethodBody(mm);
             } catch(Exception ex) {
                 throw new RuntimeException(ex);
             }

             String className = _entryPoint.getClassModel().getClassWeAreModelling().toString();
             className = className.split(" ")[1];
             String getPairsLine = className+"__getOutputPairsPerInput(this)";

             String funcDeclString = removePreviousLine();
             while(funcDeclString.indexOf("Kernel__write") == -1) {
                 funcDeclString = removePreviousLine();
             }
             write(funcDeclString);

             String arguments = funcDeclString.substring(funcDeclString.indexOf("(")+1);
             arguments = arguments.substring(0, arguments.indexOf(")"));
             String[] argTokens = arguments.split(",");

             if(outputValType.equals("svec")) {
                 write("   int index = atomic_add(this->memIncr, 1);\n");
                 write("   if (index >= this->outputLength) {\n");
                 write("      this->output_nWrites[this->iter] = -1;\n");
                 write("      return 0;\n");
                 write("   } else {\n");
                 write("      int pastWrites = this->output_nWrites[this->iter]++;\n");
                 write("      this->outputValIntLookAsideBuffer[index] = valIndices - this->outputValIndices;\n");
                 write("      this->outputValDoubleLookAsideBuffer[index] = valVals - this->outputValVals;\n");
                 write("      this->outputValLengthBuffer[index] = len;\n");
                 for(int i = 1; i < argTokens.length; i++) {
                     if(argTokens[i].indexOf("key") != -1) {
                         write("   ");
                         hadoopOutputWrite(argTokens[i]);
                     }
                 }
                 write("      return 1;\n");
                 write("   }\n");
                 write("}\n\n");
             } else {
                 write("   int index;\n");
                 write("   int pastWrites = this->output_nWrites[this->iter]++;\n");
                 write("   if (this->reservedOffset >= 0 ) {\n");
                 write("      index = this->reservedOffset;\n");
                 write("      this->reservedOffset = -1;\n");
                 write("   } else {\n");
                 if(isMapWrite) {
                     write("      index = (this->nPairs * pastWrites) + (this->iter);\n");
                     write("      if(this->isGPU == 0) {\n");
                     write("         index = ((this->iter) * "+getPairsLine+") + pastWrites;\n");
                    write("      }\n");
                 } else {
                     write("      if(this->isGPU > 0) {\n");
                     write("         index = (this->nKeys * pastWrites) + (this->iter);\n");
                     write("      } else {\n");
                     write("         index = (this->iter) * "+getPairsLine+" + pastWrites;\n");
                     write("      }\n");
                 }
                 write("   }\n");

                 for(int i = 1; i < argTokens.length; i++) {
                     hadoopOutputWrite(argTokens[i]);
                 }
                 write("   return 1;\n");
                 write("}\n\n");
             }


         } else if (isCallMap) {
             writeMethodBody(mm);

             String closeBrace = removePreviousLine();
             String returnStmt = removePreviousLine();
             String mapCall = removePreviousLine();
             String callMapSig = removePreviousLine();

             // System.out.println("mapCall=\""+mapCall+"\"");

             int openingIndex = mapCall.indexOf('(') + 1;
             int closingIndex = mapCall.lastIndexOf(')') - 1;
             String transformedArgs = mapCall.substring(openingIndex, closingIndex).replace("3", "this->iter");
             mapCall = mapCall.substring(0, openingIndex)+transformedArgs+mapCall.substring(closingIndex);

             if(inputValType.equals("svec")) {
                 String arguments = mapCall.substring(mapCall.indexOf("("));
                 arguments = arguments.substring(0, arguments.lastIndexOf(")"));
                 String[] splitArgs = arguments.split(",");
                 // System.out.println("Arguments:");
                 // for(int i =0 ; i < splitArgs.length; i++) {
                 //     System.out.println("   "+splitArgs[i]);
                 // }
                 StringBuffer rebuildCall = new StringBuffer();
                 rebuildCall.append(mapCall.substring(0, mapCall.indexOf("(")));

                 for(int i = 0; i < splitArgs.length-3; i++) {
                     rebuildCall.append(splitArgs[i]);
                     rebuildCall.append(", ");
                 }

                 if (this.getStrided()) {
                     rebuildCall.append("this->inputValIndices + (this->iter), ");
                 } else {
                     rebuildCall.append("this->inputValIndices + (this->inputValLookAsideBuffer[this->iter]), ");
                 }

                 if (this.getStrided()) {
                     rebuildCall.append("this->inputValVals + (this->iter), ");
                 } else {
                     rebuildCall.append("this->inputValVals + (this->inputValLookAsideBuffer[this->iter]), ");
                 }
                 rebuildCall.append("((this->iter == this->nPairs-1 ? "+
                         "this->individualInputValsCount : this->inputValLookAsideBuffer[this->iter+1]) "+
                         "- this->inputValLookAsideBuffer[this->iter])");

                 rebuildCall.append(");\n");
                 mapCall = rebuildCall.toString();
             }

             write(callMapSig);
             write(mapCall);
             write(returnStmt);
             write(closeBrace);

         } else if (isCallReduce) {
             writeMethodBody(mm);

             String line = removePreviousLine();
             while(line.indexOf("__callReduce") == -1) {
                 line = removePreviousLine();
             }
             write(line);
             write("   this->currentValueIndex = startOffset;\n");
             write("   this->currentStartingValueIndex = startOffset;\n");
             write("   this->currentNumValues = stopOffset - startOffset;\n");
             if(inputValType.equals("svec")) {
                write("   this->currentNumAuxValues = this->iter == this->nKeys-1 ?\n");
                write("      (this->individualInputValsCount - this->inputValLookAsideBuffer[startOffset]) :\n");
                write("      (this->inputValLookAsideBuffer[stopOffset] - this->inputValLookAsideBuffer[startOffset]);\n");
             }
             write("   "+enclosingClassName+"__reduce(this, ");

             if (inputKeyType.equals("pair")) {
                 write("this->inputKeys1[this->iter], this->inputKeys2[this->iter]");
             } else if (inputKeyType.equals("upair")) {
                 write("this->inputKeyIds[this->iter], this->inputKeys1[this->iter], this->inputKeys2[this->iter]");
             } else if (inputKeyType.equals("svec")) {
                 throw new RuntimeException("Invalid input key type svec");
             } else {
                 write("this->inputKeys[this->iter]");
             }

             write(");");
             newLine();
             write("   return;");
             newLine();
             write("}");
             newLine();
             newLine();

             //reduceCall = reduceCall.replaceAll("3", "this->iter");

             //write(reduceCall);
             //write(returnStmt);
             //write(closeBrace);
         } else if(isReserveOutput) {
             writeMethodBody(mm);

             // Should be defunct now
             String line = removePreviousLine();
             while(line.indexOf("reserveOutput") == -1) {
                 line = removePreviousLine();
             }
             write(line);

             if(outputValType.equals("svec")) {

                 write("   this->reservedAuxOffset = atomic_add(this->memAuxIncr, len);\n");
                 write("   if(this->reservedAuxOffset+len <= this->outputAuxLength) {\n");
                 write("      this->reservedOffset = atomic_add(this->memIncr, 1);\n");
                 write("   }\n");
                 write("   if (this->reservedAuxOffset+len <= this->outputAuxLength && this->reservedOffset < this->outputLength) {\n");
                 write("      this->outputValLookAsideBuffer[this->reservedOffset] = this->reservedAuxOffset;\n");
                 write("      this->outputValLengthBuffer[this->reservedOffset] = len;\n");
                 write("      return 1;\n");
                 write("   } else {\n");
                 write("      this->output_nWrites[this->iter] = -1;\n");
                 write("      return 0;\n");
                 write("   }\n");
             } else {
                 write("   this->reservedOffset = atomic_add(this->memIncr, 1);\n");
                 write("   if (this->reservedOffset >= this->outputLength) {\n");
                 write("       this->output_nWrites[this->iter] = -1;\n");
                 write("   }\n");
                 write("   return this->reservedOffset < this->outputLength;\n");
             }
             write("}\n");
         } else if(isAccessOutput) {
             writeMethodBody(mm);

             // Should be defunct now
             String line = removePreviousLine();
             while(line.indexOf("accessOutput") == -1) {
                 line = removePreviousLine();
             }
             write(line);

             write("   this->outputValIndices[this->reservedAuxOffset + index] = ind;\n");
             write("   this->outputValVals[this->reservedAuxOffset + index] = val;\n");

             write("}\n");
         } else if(isGetGlobalIndices) {
             //writeMethodBody(mm);

             //String line = removePreviousLine();
             //while(line.indexOf("getGlobalIndices") == -1) {
             //    line = removePreviousLine();
             //}
             //write(line);
             write("\n{\n");
             write("   return this->globalsInd + this->globalIndices[gid];\n");
             write("}\n");
         } else if(isGetGlobalVals) {
             //writeMethodBody(mm);

             //String line = removePreviousLine();
             //while(line.indexOf("getGlobalVals") == -1) {
             //    line = removePreviousLine();
             //}
             //write(line);
             write("\n{\n");
             write("   return this->globalsVal + this->globalIndices[gid];\n");
             write("}\n");
         } else if(isInputVectorLength) {
             write("\n{\n");
             write("   int start = this->inputValLookAsideBuffer[vid];\n");

             if(hadoopType == HADOOPTYPE.MAPPER) {
                 write("   int end = vid == this->nPairs-1 ? this->individualInputValsCount : this->inputValLookAsideBuffer[vid+1];\n");
                 write("   return end - start;\n");
             } else if (hadoopType == HADOOPTYPE.REDUCER) {
                 write("   int end = vid == this->nVals-1 ? this->individualInputValsCount : this->inputValLookAsideBuffer[vid+1];\n");
                 
             } else {
                 throw new RuntimeException("Unknown hadoop type");
             }
             write("   return end-start;\n");
             write("}\n\n");
         } else if(isAllocInt) {
             write("\n{\n");
             write("   int offset = atomic_add(this->memAuxIntIncr, len);\n");
             write("   if (offset + len > this->outputAuxLength) {\n");
             write("      return NULL;\n");
             write("   }\n");
             write("   return this->outputValIndices + offset;\n");
             write("}\n\n");
         } else if(isAllocDouble) {
             write("\n{\n");
             write("   int offset = atomic_add(this->memAuxDoubleIncr, len);\n");
             write("   if (offset + len > this->outputAuxLength) {\n");
             write("      return NULL;\n");
             write("   }\n");
             write("   return this->outputValVals + offset;\n");
             write("}\n\n");
         } else if(isMap) {
             writeMethodBody(mm);

             //List<String> buffer = new ArrayList<String>();
             //String line = removePreviousLine();
             //while(line.indexOf("map(") == -1) {
             //    buffer.add(line);
             //    line = removePreviousLine();
             //}
             //write(line);
             //for(int i = buffer.size()-1; i >= 0; i--) {
             //    write(buffer.get(i));
             //}
         } else if(isReduce) {
             writeMethodBody(mm);

             List<String> buffer = new ArrayList<String>();
             String line = removePreviousLine();
             while(line.indexOf("reduce(") == -1) {
                 buffer.add(line);
                 line = removePreviousLine();
             }
             String[] tokens = line.split(" ");
             StringBuffer rebuild = new StringBuffer();
             for(int i = 0; i < tokens.length-3; i++) {
                 rebuild.append(tokens[i]);
                 rebuild.append(" ");
             }
             String finalArg = tokens[tokens.length-3];
             finalArg = finalArg.substring(0, finalArg.length()-1);
             rebuild.append(finalArg);
             rebuild.append("){\n");
             write(rebuild.toString());
             for(int i = buffer.size()-1; i >= 0; i--) {
                 write(buffer.get(i));
             }
         } else {
             writeMethodBody(mm);
         }
         newLine();
      }

      boolean isMapRun = false;
      boolean isReduceRun = false;
      if(_entryPoint.getMethodModel().getName().indexOf(mapreducePrefix) == 0) {
         if(_entryPoint.getMethodModel().getName().indexOf(mapperRunPost) != -1) {
            isMapRun = true;
         } else if(_entryPoint.getMethodModel().getName().indexOf(reducerRunPost) != -1) {
            isReduceRun = true;
         }
      }

      write("__kernel void " + _entryPoint.getMethodModel().getSimpleName() + "(");

      in();
      boolean first = true;
      for (final String line : argLines) {

         if (first) {
            first = false;
         } else {
            write(", ");
         }

         newLine();
         write(line);
      }

      if (first) {
         first = false;
      } else {
         write(", ");
      }
      newLine();
      write("int passid");
      out();
      newLine();
      write("){");
      in();
      newLine();
      writeln("This thisStruct;");
      writeln("This* this=&thisStruct;");
      for (final String line : assigns) {
         write(line);
         writeln(";");
      }
      write("this->passid = passid");
      writeln(";");
      write("this->reservedOffset = -1");
      writeln(";");

       if(isMapRun) {

          writeMethodBody(_entryPoint.getMethodModel());
          Stack<String> removedStrings = new Stack<String>();
          String lastLine = removePreviousLine();
          while(lastLine.indexOf("for (int iter = start; iter<end; iter = iter + increment){") == -1) {
              removedStrings.push(lastLine);
              lastLine = removePreviousLine();
          }
          write("      for (this->iter = start; this->iter < end; this->iter += increment){\n");

          while(!removedStrings.empty()) {
              write(removedStrings.pop().replace("iter", "(this->iter)"));
          }

       } else if(isReduceRun) {
          writeMethodBody(_entryPoint.getMethodModel());
          Stack<String> removedStrings = new Stack<String>();
          String lastLine = removePreviousLine();
          while(lastLine.indexOf("for (int iter = start; iter<end; iter = iter + increment){") == -1) {
              removedStrings.push(lastLine);
              lastLine = removePreviousLine();
          }
          write("      for (this->iter = start; this->iter<end; this->iter += increment){\n");

          while(!removedStrings.empty()) {
              write(removedStrings.pop().replace("iter", "(this->iter)"));
          }

       } else {
          writeMethodBody(_entryPoint.getMethodModel());
       }
      out();
      newLine();
      writeln("}");
      out();
   }

   @Override public void writeThisRef() {
      write("this->");
   }

   @Override public void writeInstruction(Instruction _instruction) throws CodeGenException {
      if ((_instruction instanceof I_IUSHR) || (_instruction instanceof I_LUSHR)) {
         final BinaryOperator binaryInstruction = (BinaryOperator) _instruction;
         final Instruction parent = binaryInstruction.getParentExpr();
         boolean needsParenthesis = true;

         if (parent instanceof AssignToLocalVariable) {
            needsParenthesis = false;
         } else if (parent instanceof AssignToField) {
            needsParenthesis = false;
         } else if (parent instanceof AssignToArrayElement) {
            needsParenthesis = false;
         }
         if (needsParenthesis) {
            write("(");
         }

         if (binaryInstruction instanceof I_IUSHR) {
            write("((unsigned int)");
         } else {
            write("((unsigned long)");
         }
         writeInstruction(binaryInstruction.getLhs());
         write(")");
         write(" >> ");
         writeInstruction(binaryInstruction.getRhs());

         if (needsParenthesis) {
            write(")");
         }
      } else {
         super.writeInstruction(_instruction);
      }
   }

   public static class StringList {
       private LinkedList<String> strings = new LinkedList<String>();

       public void append(String s) {
           strings.add(s);
       }

       public void insertAfter(String toInsert, String after) {
           int index = 0;
           while(strings.get(index).indexOf(after) != 0) {
               index = index + 1;
           }
           strings.add(index+1, toInsert);
       }

       @Override
       public String toString() {
           StringBuilder builder = new StringBuilder();
           for(String s : strings) {
               builder.append(s);
           }
           return builder.toString();
       }

       public List<String> getList() {
           return strings;
       }

       public String removePreviousLine() {
           String acc = "";
           String lastString = strings.pollLast();
           while(lastString.length() == 0) {
               lastString = strings.pollLast();
           }

           boolean appendNewLine = false;
           if(lastString.charAt(lastString.length()-1) == '\n') {
               appendNewLine = true;
               lastString = lastString.substring(0, lastString.length()-1);
           }

           while(lastString.lastIndexOf('\n') == -1 && strings.size() > 1) {
              acc = lastString + acc;
              lastString = strings.pollLast();
           }

           if(lastString.lastIndexOf('\n') != lastString.length()-1) {
               int lastIndex = lastString.lastIndexOf('\n');

               strings.add(lastString.substring(0, lastIndex+1));
               acc = lastString.substring(lastIndex+1)+acc;
           } else {
               strings.add(lastString);
           }
           if(appendNewLine) {
               acc = acc + "\n";
           }
           return acc;
       }
   }

   public static class OpenCLKernelWriter extends KernelWriter {
       private final StringList openCLStringBuilder;

       public OpenCLKernelWriter() {
           this.openCLStringBuilder = new StringList();
       }

       @Override public void write(String _string) {
           openCLStringBuilder.append(_string);
       }

       @Override protected String removePreviousLine() {
           return openCLStringBuilder.removePreviousLine();
       }

       @Override public String toString() {
           return openCLStringBuilder.toString();
       }
   }

   public static String writeToString(Entrypoint _entrypoint,
           Entrypoint _entrypointcopy, boolean isGPU, boolean enableStrided)
               throws CodeGenException {
      final OpenCLKernelWriter tmpOpenCLWriter = new OpenCLKernelWriter();

      final boolean VERBOSE = false;
      try {
         tmpOpenCLWriter.write(_entrypoint);
      } catch (final CodeGenException codeGenException) {
         throw new RuntimeException(codeGenException);
      }

      HadoopTypes types = KernelWriter.types;
      if (VERBOSE) {
          System.out.println(types.toString());
          System.out.println("Running on "+(isGPU ? "GPU" : "CPU"));
      }

      // if (true) {
      if (!enableStrided || types.hadoopType() != HADOOPTYPE.MAPPER ||
              !types.inputValType().equals("svec") ||
              !isGPU) {
          return tmpOpenCLWriter.toString();
      }

      HashMap<String, MethodArgumentList> methodArgs = tmpOpenCLWriter.getMethodArgs();
      if (VERBOSE) {
          System.out.println("Uncovered arguments for "+
                  methodArgs.size()+" methods:");
          for(String methodName : methodArgs.keySet()) {
              MethodArgumentList args = methodArgs.get(methodName);
              System.out.println("  "+args.toString());
          }
      }

      tmpOpenCLWriter.resolveAliases();

      Set<LocalVar> allStrided = new HashSet<LocalVar>();
      Set<LocalVar> allUnstrided = new HashSet<LocalVar>();

      Set<VarAlias> aliases = tmpOpenCLWriter.getAliases();
      for(VarAlias al : aliases) {
          LocalVar var = al.getBeingPassed();
          if (var.isMapLocal()) {
              if (var.isMapParameter(methodArgs)) {
                  allStrided.add(var);
              } else {
                  allUnstrided.add(var);
              }
          }
      }

      

      boolean changed;
      do {
          changed = false;

          for(VarAlias al : aliases) {
              if (allStrided.contains(al.getBeingPassed()) && 
                      !allStrided.contains(al.getBeingPassedAs())) {
                  allStrided.add(al.getBeingPassedAs());
                  changed = true;
              } 
              if (allUnstrided.contains(al.getBeingPassed()) &&
                      !allUnstrided.contains(al.getBeingPassedAs())) {
                  allUnstrided.add(al.getBeingPassedAs());
                  changed = true;
              }
          }
      } while(changed);

      Set<LocalVar> intersection = new HashSet<LocalVar>(allStrided);
      intersection.retainAll(allUnstrided);
      if (intersection.isEmpty()) {
          if (VERBOSE) {
              System.out.println("Intersection Empty!");
          }
      } else {
          if (VERBOSE) {
              System.out.println("Intersection:");
              for(LocalVar both : intersection) {
                  System.out.println("  "+both.toString());
              }
          }
          throw new RuntimeException("Intersection between strided and unstrided variables");
      }

      HashMap<LocalVar, Boolean> allVars = new HashMap<LocalVar, Boolean>();
      if (VERBOSE) {
          System.out.println("Uncovered "+aliases.size()+ " aliases");
      }
      for(VarAlias al : aliases) {
          allVars.put(al.getBeingPassed(), allStrided.contains(al.getBeingPassed()));
          allVars.put(al.getBeingPassedAs(), allStrided.contains(al.getBeingPassedAs()));

          if (VERBOSE) {
              System.out.println("  "+al.toString());
          }
      }

      for (String methodName : methodArgs.keySet()) {
          final String mapSuffix = "__map";
          int index = methodName.indexOf(mapSuffix);
          if (index == -1 || index != methodName.length() - mapSuffix.length()) {
              continue;
          }
          MethodArgumentList args = methodArgs.get(methodName);
          for (int i = 0; i < args.size(); i++) {
              String varName = args.getArg(i);
              allVars.put(new LocalVar(methodName, varName), true);
          }
          break;
      }

      if (VERBOSE) {
          System.out.println("Variables:");
          for (LocalVar v : allVars.keySet()) {
              System.out.println("  "+v.toString()+" -> "+allVars.get(v));
          }
      }

      final OpenCLKernelWriter openCLWriter = new OpenCLKernelWriter();
      openCLWriter.setAllVars(allVars);
      openCLWriter.setAliases(aliases);
      openCLWriter.setMethodArgs(methodArgs);
      openCLWriter.setStrided(enableStrided);

      try {
         openCLWriter.write(_entrypointcopy);
      } catch (final CodeGenException codeGenException) {
         throw new RuntimeException(codeGenException);
      }

      // String result = openCLWriter.toString();
      // System.out.println("Got result with length "+result.length());
      // System.out.println("---------------------------");
      // System.out.println(result);
      // System.out.println("---------------------------");
      return openCLWriter.toString();
   }

   public static enum HADOOPTYPE {
       UNKNOWN, MAPPER, REDUCER
   }

   public static class HadoopTypes {
       private final HADOOPTYPE hadoopType;
       private final String inputKeyType;
       private final String inputValType;
       private final String outputKeyType;
       private final String outputValType;

       public HadoopTypes(HADOOPTYPE hadoopType,
               String ikt, String ivt, String okt, String ovt) {
           this.hadoopType = hadoopType;
           this.inputKeyType = ikt;
           this.inputValType = ivt;
           this.outputKeyType = okt;
           this.outputValType = ovt;
       }

       public String inputKeyType() { return this.inputKeyType; }
       public String inputValType() { return this.inputValType; }
       public String outputKeyType() { return this.outputKeyType; }
       public String outputValType() { return this.outputValType; }
       public HADOOPTYPE hadoopType() { return this.hadoopType; }
       @Override
       public String toString() {
           String hadoopStr;
           if (hadoopType == HADOOPTYPE.MAPPER) {
               hadoopStr = "MAPPER";
           } else if (hadoopType == HADOOPTYPE.REDUCER) {
               hadoopStr = "REDUCER";
           } else {
               hadoopStr = "UNKNOWN";
           }
           return hadoopStr+"("+inputKeyType+", "+inputValType+") -> ("+
               outputKeyType+", "+outputValType+")";
       }
   }
}
