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

import java.util.Stack;
import java.util.Set;
import java.util.HashSet;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import com.amd.aparapi.internal.exception.CodeGenException;
import com.amd.aparapi.internal.instruction.BranchSet;
import com.amd.aparapi.internal.instruction.Instruction;
import com.amd.aparapi.internal.instruction.BranchSet.CompoundLogicalExpressionNode;
import com.amd.aparapi.internal.instruction.BranchSet.LogicalExpressionNode;
import com.amd.aparapi.internal.instruction.BranchSet.SimpleLogicalExpressionNode;
import com.amd.aparapi.internal.instruction.InstructionSet.AccessArrayElement;
import com.amd.aparapi.internal.instruction.InstructionSet.AccessField;
import com.amd.aparapi.internal.instruction.InstructionSet.AccessInstanceField;
import com.amd.aparapi.internal.instruction.InstructionSet.AccessLocalVariable;
import com.amd.aparapi.internal.instruction.InstructionSet.AssignToArrayElement;
import com.amd.aparapi.internal.instruction.InstructionSet.AssignToField;
import com.amd.aparapi.internal.instruction.InstructionSet.AssignToInstanceField;
import com.amd.aparapi.internal.instruction.InstructionSet.AssignToLocalVariable;
import com.amd.aparapi.internal.instruction.InstructionSet.BinaryOperator;
import com.amd.aparapi.internal.instruction.InstructionSet.Branch;
import com.amd.aparapi.internal.instruction.InstructionSet.ByteCode;
import com.amd.aparapi.internal.instruction.InstructionSet.CastOperator;
import com.amd.aparapi.internal.instruction.InstructionSet.CloneInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.CompositeArbitraryScopeInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.CompositeDoWhileInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.CompositeEmptyLoopInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.CompositeForEclipseInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.CompositeForSunInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.CompositeIfElseInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.CompositeIfInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.CompositeInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.CompositeWhileInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.ConditionalBranch16;
import com.amd.aparapi.internal.instruction.InstructionSet.Constant;
import com.amd.aparapi.internal.instruction.InstructionSet.FieldArrayElementAssign;
import com.amd.aparapi.internal.instruction.InstructionSet.FieldArrayElementIncrement;
import com.amd.aparapi.internal.instruction.InstructionSet.I_ALOAD_0;
import com.amd.aparapi.internal.instruction.InstructionSet.I_AALOAD;
import com.amd.aparapi.internal.instruction.InstructionSet.I_ARRAYLENGTH;
import com.amd.aparapi.internal.instruction.InstructionSet.I_IFNONNULL;
import com.amd.aparapi.internal.instruction.InstructionSet.I_IFNULL;
import com.amd.aparapi.internal.instruction.InstructionSet.I_IINC;
import com.amd.aparapi.internal.instruction.InstructionSet.I_POP;
import com.amd.aparapi.internal.instruction.InstructionSet.I_ACONST_NULL;
import com.amd.aparapi.internal.instruction.InstructionSet.If;
import com.amd.aparapi.internal.instruction.InstructionSet.IfUnary;
import com.amd.aparapi.internal.instruction.InstructionSet.IncrementInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.InlineAssignInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.MethodCall;
import com.amd.aparapi.internal.instruction.InstructionSet.MultiAssignInstruction;
import com.amd.aparapi.internal.instruction.InstructionSet.Return;
import com.amd.aparapi.internal.instruction.InstructionSet.UnaryOperator;
import com.amd.aparapi.internal.instruction.InstructionSet.VirtualMethodCall;
import com.amd.aparapi.internal.model.Entrypoint;
import com.amd.aparapi.internal.model.MethodModel;
import com.amd.aparapi.internal.model.ClassModel.ConstantPool.MethodEntry;
import com.amd.aparapi.internal.model.ClassModel.LocalVariableInfo;

/**
 * Base abstract class for converting <code>Aparapi</code> IR to text.<br/>
 * 
 *   
 * @author gfrost
 *
 */

public abstract class BlockWriter{

   private boolean enableStrided = false;
   private HashMap<LocalVar, Boolean> allVars = null;
   private Set<VarAlias> aliases = new HashSet<VarAlias>();
   private HashMap<String, MethodArgumentList> methodsToArgs =
       new HashMap<String, MethodArgumentList>();
   protected String currentMethodBody;
   private boolean aliasesReadOnly = false;

   protected void setStrided(boolean set) {
       this.enableStrided = set;
   }
   protected boolean getStrided() { return this.enableStrided; }
   protected void setAliases(Set<VarAlias> aliases) {
       this.aliasesReadOnly = true;
       this.aliases = aliases;
   }
   protected Set<VarAlias> getAliases() { return this.aliases; }

   protected void setMethodArgs(HashMap<String, MethodArgumentList> methodArgs) {
       this.aliasesReadOnly = true;
       this.methodsToArgs = methodArgs;
   }
   protected HashMap<String, MethodArgumentList> getMethodArgs() {
       return this.methodsToArgs;
   }

   protected void setAllVars(HashMap<LocalVar, Boolean> allVars) {
       this.aliasesReadOnly = true;
       this.allVars = allVars;
   }
   protected HashMap<LocalVar, Boolean> getAllVars() { return this.allVars; }

   protected void resolveAliases() {
       for(VarAlias alias : aliases) {
          alias.getBeingPassedAs().resolve(methodsToArgs);
       }
   }

   protected void addAlias(VarAlias newAlias) {
       if (!aliasesReadOnly) {
           this.aliases.add(newAlias);
       }
   }

   protected void addMethodArgs(String methodName, MethodArgumentList methodArgs) {
       if (!aliasesReadOnly) {
           this.methodsToArgs.put(methodName, methodArgs);
       }
   }

   public final static String arrayLengthMangleSuffix = "__javaArrayLength";
   public final static String arrayDimMangleSuffix = "__javaArrayDimension";

   public abstract void write(String _string);

   public void writeln(String _string) {
      write(_string);
      newLine();
   }

   public int indent = 0;

   public void in() {
      indent++;
   }

   public void out() {
      indent--;
   }

   public void newLine() {
      write("\n");
      for (int i = 0; i < indent; i++) {
         write("   ");
      }
   }

   public void writeConditionalBranch16(ConditionalBranch16 _branch16, boolean _invert) throws CodeGenException {

      if (_branch16 instanceof If) {
         final If iff = (If) _branch16;

         writeInstruction(iff.getLhs());
         write(_branch16.getOperator().getText(_invert));
         writeInstruction(iff.getRhs());
      } else if (_branch16 instanceof I_IFNULL) {
         final I_IFNULL iff = (I_IFNULL) _branch16;
         writeInstruction(iff.getFirstChild());

         if (_invert) {
            write(" != NULL");
         } else {
            write(" == NULL");
         }

      } else if (_branch16 instanceof I_IFNONNULL) {
         final I_IFNONNULL iff = (I_IFNONNULL) _branch16;
         writeInstruction(iff.getFirstChild());

         if (_invert) {
            write(" == NULL");
         } else {
            write(" != NULL");
         }
      } else if (_branch16 instanceof IfUnary) {
         final IfUnary branch16 = (IfUnary) _branch16;
         final Instruction comparison = branch16.getUnary();
         final ByteCode comparisonByteCode = comparison.getByteCode();
         final String comparisonOperator = _branch16.getOperator().getText(_invert);

         switch (comparisonByteCode) {
            case FCMPG:
            case DCMPG:
            case FCMPL:
            case DCMPL:
               writeInstruction(comparison.getFirstChild());
               write(comparisonOperator);
               writeInstruction(comparison.getLastChild());
               break;
            default:
               writeInstruction(comparison);
               write(comparisonOperator);
               write("0");
         }
      }
   }

   public void writeComposite(CompositeInstruction instruction) throws CodeGenException {
      if (instruction instanceof CompositeArbitraryScopeInstruction) {
         newLine();

         writeBlock(instruction.getFirstChild(), null);
      } else if (instruction instanceof CompositeIfInstruction) {
         newLine();
         write("if (");
         final Instruction blockStart = writeConditional(instruction.getBranchSet());

         write(")");
         writeBlock(blockStart, null);
      } else if (instruction instanceof CompositeIfElseInstruction) {
         newLine();
         write("if (");
         final Instruction blockStart = writeConditional(instruction.getBranchSet());
         write(")");
         Instruction elseGoto = blockStart;
         while (!(elseGoto.isBranch() && elseGoto.asBranch().isUnconditional())) {
            elseGoto = elseGoto.getNextExpr();
         }
         writeBlock(blockStart, elseGoto);
         write(" else ");
         writeBlock(elseGoto.getNextExpr(), null);
      } else if (instruction instanceof CompositeForSunInstruction) {
         newLine();
         write("for (");
         Instruction topBranch = instruction.getFirstChild();
         if (topBranch instanceof AssignToLocalVariable) {
            writeInstruction(topBranch);
            topBranch = topBranch.getNextExpr();
         }
         write("; ");
         final BranchSet branchSet = instruction.getBranchSet();
         final Instruction blockStart = writeConditional(branchSet);

         final Instruction lastGoto = instruction.getLastChild();

         if (branchSet.getFallThrough() == lastGoto) {
            // empty body no delta!
            write(";){}");
         } else {
            final Instruction delta = lastGoto.getPrevExpr();
            write("; ");
            if (!(delta instanceof CompositeInstruction)) {
               writeInstruction(delta);
               write(")");
               writeBlock(blockStart, delta);
            } else {
               write("){");
               in();
               writeSequence(blockStart, delta);

               newLine();
               writeSequence(delta, delta.getNextExpr());
               out();
               newLine();
               write("}");

            }
         }

      } else if (instruction instanceof CompositeWhileInstruction) {
         newLine();
         write("while (");
         final BranchSet branchSet = instruction.getBranchSet();
         final Instruction blockStart = writeConditional(branchSet);
         write(")");
         final Instruction lastGoto = instruction.getLastChild();
         writeBlock(blockStart, lastGoto);

      } else if (instruction instanceof CompositeEmptyLoopInstruction) {
         newLine();
         write("for (");
         Instruction topBranch = instruction.getFirstChild();
         if (topBranch instanceof AssignToLocalVariable) {
            writeInstruction(topBranch);
            topBranch = topBranch.getNextExpr();
         }
         write("; ");
         writeConditional(instruction.getBranchSet());
         write(";){}");

      } else if (instruction instanceof CompositeForEclipseInstruction) {
         newLine();
         System.err.println("Found CompositeForEclipseInstruction");
         write("for (");
         Instruction topGoto = instruction.getFirstChild();
         if (topGoto instanceof AssignToLocalVariable) {
            writeInstruction(topGoto);
            topGoto = topGoto.getNextExpr();
         }
         write("; ");
         Instruction last = instruction.getLastChild();
         while (last.getPrevExpr().isBranch()) {
            last = last.getPrevExpr();
         }
         writeConditional(instruction.getBranchSet(), true);
         write("; ");
         final Instruction delta = last.getPrevExpr();
         if (!(delta instanceof CompositeInstruction)) {
            writeInstruction(delta);
            write(")");
            writeBlock(topGoto.getNextExpr(), delta);
         } else {
            write("){");
            in();
            writeSequence(topGoto.getNextExpr(), delta);

            newLine();
            writeSequence(delta, delta.getNextExpr());
            out();
            newLine();
            write("}");

         }

      } else if (instruction instanceof CompositeDoWhileInstruction) {
         newLine();
         write("do");
         Instruction blockStart = instruction.getFirstChild();
         Instruction blockEnd = instruction.getLastChild();
         writeBlock(blockStart, blockEnd);
         write("while(");
         writeConditional(((CompositeInstruction) instruction).getBranchSet(), true);
         write(");");
         newLine();
      }
   }

   public void writeSequence(Instruction _first, Instruction _last) throws CodeGenException {

      for (Instruction instruction = _first; instruction != _last; instruction = instruction.getNextExpr()) {
         if (instruction instanceof CompositeInstruction) {
            writeComposite((CompositeInstruction) instruction);
         } else if (!instruction.getByteCode().equals(ByteCode.NONE)) {
            newLine();
            writeInstruction(instruction);
            write(";");

         }
      }

   }

   public void writeBlock(Instruction _first, Instruction _last) throws CodeGenException {
      write("{");
      in();
      writeSequence(_first, _last);
      out();
      newLine();

      write("}");
   }

   public Instruction writeConditional(BranchSet _branchSet) throws CodeGenException {
      return (writeConditional(_branchSet, false));
   }

   public Instruction writeConditional(BranchSet _branchSet, boolean _invert) throws CodeGenException {

      final LogicalExpressionNode logicalExpression = _branchSet.getLogicalExpression();
      if (!_invert) {
         logicalExpression.invert();
      }
      write(logicalExpression);
      return (_branchSet.getLast().getNextExpr());
   }

   public void write(LogicalExpressionNode _node) throws CodeGenException {
      if (_node instanceof SimpleLogicalExpressionNode) {
         final SimpleLogicalExpressionNode sn = (SimpleLogicalExpressionNode) _node;

         writeConditionalBranch16((ConditionalBranch16) sn.getBranch(), sn.isInvert());
      } else {
         final CompoundLogicalExpressionNode ln = (CompoundLogicalExpressionNode) _node;
         boolean needParenthesis = false;
         final CompoundLogicalExpressionNode parent = (CompoundLogicalExpressionNode) ln.getParent();
         if (parent != null) {
            if (!ln.isAnd() && parent.isAnd()) {
               needParenthesis = true;
            }
         }
         if (needParenthesis) {

            write("(");
         }
         write(ln.getLhs());
         write(ln.isAnd() ? " && " : " || ");
         write(ln.getRhs());
         if (needParenthesis) {

            write(")");
         }
      }
   }

   public String convertType(String _typeDesc, boolean useClassModel) {
      return (_typeDesc);
   }

   public String convertCast(String _cast) {
      // Strip parens off cast
      //System.out.println("cast = " + _cast);
      final String raw = convertType(_cast.substring(1, _cast.length() - 1), false);
      return ("(" + raw + ")");
   }

   public void writeInstruction(Instruction _instruction) throws CodeGenException {
      if (_instruction instanceof CompositeIfElseInstruction) {
         write("(");
         final Instruction lhs = writeConditional(((CompositeInstruction) _instruction).getBranchSet());
         write(")?");
         writeInstruction(lhs);
         write(":");
         writeInstruction(lhs.getNextExpr().getNextExpr());
      } else if (_instruction instanceof CompositeInstruction) {
         writeComposite((CompositeInstruction) _instruction);

      } else if (_instruction instanceof AssignToLocalVariable) {
         final AssignToLocalVariable assignToLocalVariable = (AssignToLocalVariable) _instruction;

         final LocalVariableInfo localVariableInfo = assignToLocalVariable.getLocalVariableInfo();
         if (localVariableInfo == null) {
            throw new CodeGenException("outOfScope" + _instruction.getThisPC() + " = ");
         } else {
            int numOperands = 0;
            for (Instruction operand = _instruction.getFirstChild(); operand != null; operand = operand.getNextExpr()) {
               numOperands++;
            }

            boolean allocCall = false;

            if(numOperands == 1 && _instruction.getFirstChild() instanceof MethodCall) {
                final MethodCall call = (MethodCall)_instruction.getFirstChild();
                final MethodEntry methodEntry = call.getConstantPoolMethodEntry();
                String methodName = methodEntry.getNameAndTypeEntry().getNameUTF8Entry().getUTF8();
                if(methodName.indexOf("alloc") == 0) {
                    allocCall = true;
                }
            }

            if(allocCall) {
               final MethodCall call = (MethodCall)_instruction.getFirstChild();
               final MethodEntry methodEntry = call.getConstantPoolMethodEntry();
               String methodName = methodEntry.getNameAndTypeEntry().getNameUTF8Entry().getUTF8();
               if (assignToLocalVariable.isDeclaration()) {
                  final String descriptor = localVariableInfo.getVariableDescriptor();
                  // Arrays always map to __global arrays
                  if (descriptor.startsWith("[")) {
                     write(" __global ");
                  }
                  write(convertType(descriptor, true));
                  write(localVariableInfo.getVariableName()+";");
                  newLine();
               }
               if(methodName.equals("allocInt")) {
                   write("{ __global int *tmp = ");
               } else if(methodName.equals("allocDouble")) {
                   write("{ __global double *tmp = ");
               } else if(methodName.equals("allocFloat")) {
                   write("{ __global float *tmp = ");
               } else {
                   throw new RuntimeException("Unsupported alloc method \""+methodName+"\"");
               }
               writeInstruction(_instruction.getFirstChild());
               write(";");
               newLine();
               write("if (tmp == NULL) return;");
               newLine();
               write(localVariableInfo.getVariableName()+" = tmp; }");
               newLine();

            } else {
               if (assignToLocalVariable.isDeclaration()) {
                  final String descriptor = localVariableInfo.getVariableDescriptor();
                  // Arrays always map to __global arrays
                  if (descriptor.startsWith("[")) {
                     write(" __global ");
                  }
                  write(convertType(descriptor, true));
               }

               write(localVariableInfo.getVariableName() + " = ");
               int numOps = 0;
               for (Instruction operand = _instruction.getFirstChild(); operand != null; operand = operand.getNextExpr()) {
                  writeInstruction(operand);
                  numOps++;
               }

               if (numOps == 1 && _instruction.getFirstChild() instanceof AccessLocalVariable) {
                 String variableName = ((AccessLocalVariable) _instruction.getFirstChild()).getLocalVariableInfo().getVariableName();
                 LocalVar assignedFrom = new LocalVar(this.currentMethodBody,
                     variableName);
                 LocalVar assignedTo = new LocalVar(this.currentMethodBody, localVariableInfo.getVariableName());
                 this.addAlias(new VarAlias(assignedFrom, assignedTo));
               }
            }
         }


      } else if (_instruction instanceof AssignToArrayElement) {
         final AssignToArrayElement arrayAssignmentInstruction = (AssignToArrayElement) _instruction;
         final Instruction refInstruction = arrayAssignmentInstruction.getArrayRef();
         writeInstruction(refInstruction);

         boolean strided = false;
         if (this.getStrided() && this.allVars != null &&
                 refInstruction instanceof AccessLocalVariable) {
             final AccessLocalVariable localVariableLoadInstruction =
               (AccessLocalVariable) refInstruction;
             final LocalVariableInfo localVariable =
               localVariableLoadInstruction.getLocalVariableInfo();
             if(localVariable.isArray()) {
                 String varName = localVariable.getVariableName();
                 LocalVar var = new LocalVar(this.currentMethodBody, varName);
                 Boolean obj = this.allVars.get(var);
                 /*
                  * In testing, it seems that arrays retrieved through HadoopCL globals
                  * may have null entries in allVars. They can never be strided, so we just
                  * handle that here.
                  */
                 if (obj != null && obj.booleanValue()) {
                   strided = true;
                 }
             }
         }

         write("[");
         if (strided) write("(");
         writeInstruction(arrayAssignmentInstruction.getArrayIndex());
         if (strided) write(") * this->nPairs");
         write("]");
         write(" ");
         write(" = ");
         writeInstruction(arrayAssignmentInstruction.getValue());
      } else if (_instruction instanceof AccessArrayElement) {

         //we're getting an element from an array
         //if the array is a primitive then we just return the value
         //so the generated code looks like
         //arrayName[arrayIndex];
         //but if the array is an object, or multidimensional array, then we want to return
         //a pointer to our index our position in the array.  The code will look like
         //&(arrayName[arrayIndex * this->arrayNameLen_dimension]
         //
         final AccessArrayElement arrayLoadInstruction = (AccessArrayElement) _instruction;

         //object array, get address
         if(arrayLoadInstruction instanceof I_AALOAD) {
            write("(&");
         }
         Instruction refInstruction = arrayLoadInstruction.getArrayRef();

         boolean strided = false;
         if (this.getStrided() && this.allVars != null && refInstruction instanceof AccessLocalVariable) {
            final AccessLocalVariable localVariableLoadInstruction = (AccessLocalVariable) refInstruction;
            final LocalVariableInfo localVariable = localVariableLoadInstruction.getLocalVariableInfo();
            if(localVariable.isArray()) {
                String varName = localVariable.getVariableName();
                LocalVar var = new LocalVar(this.currentMethodBody, varName);
                Boolean obj = this.allVars.get(var);
                /*
                 * In testing, it seems that arrays retrieved through HadoopCL globals
                 * may have null entries in allVars. They can never be strided, so we just
                 * handle that here.
                 */
                if (obj != null && obj.booleanValue()) {
                  strided = true;
                }
            }
         }
         writeInstruction(refInstruction);
         write("[");
         if (strided) write("(");
         Instruction indexInstruction = arrayLoadInstruction.getArrayIndex();
         // Correct place to modify array reads from mapper inputs
         writeInstruction(indexInstruction);

         //object array, find the size of each object in the array
         //for 2D arrays, this size is the size of a row.
         if(arrayLoadInstruction instanceof I_AALOAD) {
             throw new RuntimeException("I've (Max) commented this out, not sure what it does and it was breaking kernel arg setting");
            //int dim = 0;
            //Instruction load = arrayLoadInstruction.getArrayRef();
            //while(load instanceof I_AALOAD) {
            //   load = load.getFirstChild();
            //   dim++;
            //}

            //String arrayName = ((AccessInstanceField)load).getConstantPoolFieldEntry().getNameAndTypeEntry().getNameUTF8Entry().getUTF8();
            //write(" * this->" + arrayName + arrayDimMangleSuffix+dim);
         }

         if (strided) {
             write(") * this->nPairs");
         }
         write("]");

         //object array, close parentheses
         if(arrayLoadInstruction instanceof I_AALOAD) {
            write(")");
         }
      } else if (_instruction instanceof AccessField) {
         final AccessField accessField = (AccessField) _instruction;
         if (accessField instanceof AccessInstanceField) {
            Instruction accessInstanceField = ((AccessInstanceField) accessField).getInstance();
            if (accessInstanceField instanceof CloneInstruction) {
               accessInstanceField = ((CloneInstruction) accessInstanceField).getReal();
            }
            if (!(accessInstanceField instanceof I_ALOAD_0)) {
               writeInstruction(accessInstanceField);
               write(".");
            } else {
               writeThisRef();
            }
         }
         write(accessField.getConstantPoolFieldEntry().getNameAndTypeEntry().getNameUTF8Entry().getUTF8());

      } else if (_instruction instanceof I_ARRAYLENGTH) {

         //getting the length of an array.
         //if this is a primitive array, then this is trivial
         //if we're getting an object array, then we need to find what dimension
         //we're looking at
         int dim = 0;
         Instruction load = _instruction.getFirstChild();
         while(load instanceof I_AALOAD) {
             load = load.getFirstChild();
             dim++;
         }
         final AccessInstanceField child = (AccessInstanceField) load;
         final String arrayName = child.getConstantPoolFieldEntry().getNameAndTypeEntry().getNameUTF8Entry().getUTF8();
         write("this->" + arrayName + arrayLengthMangleSuffix + dim);
      } else if (_instruction instanceof AssignToField) {
         final AssignToField assignedField = (AssignToField) _instruction;

         if (assignedField instanceof AssignToInstanceField) {
            final Instruction accessInstanceField = ((AssignToInstanceField) assignedField).getInstance().getReal();

            if (!(accessInstanceField instanceof I_ALOAD_0)) {
               writeInstruction(accessInstanceField);
               write(".");
            } else {
               writeThisRef();
            }
         }
         write(assignedField.getConstantPoolFieldEntry().getNameAndTypeEntry().getNameUTF8Entry().getUTF8());
         write("=");
         writeInstruction(assignedField.getValueToAssign());
      } else if (_instruction instanceof Constant<?>) {
         final Constant<?> constantInstruction = (Constant<?>) _instruction;
         final Object value = constantInstruction.getValue();

         if (value instanceof Float) {

            final Float f = (Float) value;
            if (f.isNaN()) {
               write("NAN");
            } else if (f.isInfinite()) {
               if (f < 0) {
                  write("-");
               }
               write("INFINITY");
            } else {
               write(value.toString());
               write("f");
            }
         } else if (value instanceof Double) {

            final Double d = (Double) value;
            if (d.isNaN()) {
               write("NAN");
            } else if (d.isInfinite()) {
               if (d < 0) {
                  write("-");
               }
               write("INFINITY");
            } else {
               write(value.toString());
            }
         } else {
           if (constantInstruction instanceof I_ACONST_NULL) {
             write("NULL");
           } else {
            write(value.toString());
            if (value instanceof Long) {
               write("L");
            }
           }
         }

      } else if (_instruction instanceof AccessLocalVariable) {
         final AccessLocalVariable localVariableLoadInstruction = (AccessLocalVariable) _instruction;
         final LocalVariableInfo localVariable = localVariableLoadInstruction.getLocalVariableInfo();
         write(localVariable.getVariableName());
      } else if (_instruction instanceof I_IINC) {
         final I_IINC location = (I_IINC) _instruction;
         final LocalVariableInfo localVariable = location.getLocalVariableInfo();
         final int adjust = location.getAdjust();

         write(localVariable.getVariableName());
         if (adjust == 1) {
            write("++");
         } else if (adjust == -1) {
            write("--");
         } else if (adjust > 1) {
            write("+=" + adjust);
         } else if (adjust < -1) {
            write("-=" + (-adjust));
         }
      } else if (_instruction instanceof BinaryOperator) {
         final BinaryOperator binaryInstruction = (BinaryOperator) _instruction;
         final Instruction parent = binaryInstruction.getParentExpr();
         boolean needsParenthesis = true;

         if (parent instanceof AssignToLocalVariable) {
            needsParenthesis = false;
         } else if (parent instanceof AssignToField) {
            needsParenthesis = false;
         } else if (parent instanceof AssignToArrayElement) {
            needsParenthesis = false;
         } else {
            /**
                        if (parent instanceof BinaryOperator) {
                           BinaryOperator parentBinaryOperator = (BinaryOperator) parent;
                           if (parentBinaryOperator.getOperator().ordinal() > binaryInstruction.getOperator().ordinal()) {
                              needsParenthesis = false;
                           }
                        }
            **/
         }

         if (needsParenthesis) {
            write("(");
         }

         writeInstruction(binaryInstruction.getLhs());

         write(" " + binaryInstruction.getOperator().getText() + " ");
         writeInstruction(binaryInstruction.getRhs());

         if (needsParenthesis) {
            write(")");
         }

      } else if (_instruction instanceof CastOperator) {
         final CastOperator castInstruction = (CastOperator) _instruction;
         //  write("(");
         write(convertCast(castInstruction.getOperator().getText()));

         writeInstruction(castInstruction.getUnary());
         //    write(")");
      } else if (_instruction instanceof UnaryOperator) {
         final UnaryOperator unaryInstruction = (UnaryOperator) _instruction;
         //   write("(");
         write(unaryInstruction.getOperator().getText());

         writeInstruction(unaryInstruction.getUnary());
         //   write(")");
      } else if (_instruction instanceof Return) {

         final Return ret = (Return) _instruction;
         write("return");
         if (ret.getStackConsumeCount() > 0) {
            write("(");
            writeInstruction(ret.getFirstChild());
            write(")");
         }

      } else if (_instruction instanceof MethodCall) {
         final MethodCall methodCall = (MethodCall) _instruction;

         final MethodEntry methodEntry = methodCall.getConstantPoolMethodEntry();

         String methodName = methodEntry.getNameAndTypeEntry().getNameUTF8Entry().getUTF8();

         if(methodName.equals("write")) {
             write("if (!");
             writeMethod(methodCall, methodEntry);
             write(") return");
         } else if (methodName.equals("nValues")) {
             write("nValues(this)");
         } else {
             writeMethod(methodCall, methodEntry);
         }

      } else if (_instruction.getByteCode().equals(ByteCode.CLONE)) {
         final CloneInstruction cloneInstruction = (CloneInstruction) _instruction;
         writeInstruction(cloneInstruction.getReal());
      } else if (_instruction.getByteCode().equals(ByteCode.INCREMENT)) {
         final IncrementInstruction incrementInstruction = (IncrementInstruction) _instruction;

         if (incrementInstruction.isPre()) {
            if (incrementInstruction.isInc()) {
               write("++");
            } else {
               write("--");
            }
         }

         writeInstruction(incrementInstruction.getFieldOrVariableReference());
         if (!incrementInstruction.isPre()) {
            if (incrementInstruction.isInc()) {
               write("++");
            } else {
               write("--");
            }
         }
      } else if (_instruction.getByteCode().equals(ByteCode.MULTI_ASSIGN)) {
         final MultiAssignInstruction multiAssignInstruction = (MultiAssignInstruction) _instruction;
         AssignToLocalVariable from = (AssignToLocalVariable) multiAssignInstruction.getFrom();
         final AssignToLocalVariable last = (AssignToLocalVariable) multiAssignInstruction.getTo();
         final Instruction common = multiAssignInstruction.getCommon();
         final Stack<AssignToLocalVariable> stack = new Stack<AssignToLocalVariable>();

         while (from != last) {
            stack.push(from);
            from = (AssignToLocalVariable) ((Instruction) from).getNextExpr();
         }

         for (AssignToLocalVariable alv = stack.pop(); alv != null; alv = stack.size() > 0 ? stack.pop() : null) {

            final LocalVariableInfo localVariableInfo = alv.getLocalVariableInfo();
            if (alv.isDeclaration()) {
               write(convertType(localVariableInfo.getVariableDescriptor(), true));
            }
            if (localVariableInfo == null) {
               throw new CodeGenException("outOfScope" + _instruction.getThisPC() + " = ");
            } else {
               write(localVariableInfo.getVariableName() + " = ");
            }

         }
         writeInstruction(common);
      } else if (_instruction.getByteCode().equals(ByteCode.INLINE_ASSIGN)) {
         final InlineAssignInstruction inlineAssignInstruction = (InlineAssignInstruction) _instruction;
         final AssignToLocalVariable assignToLocalVariable = inlineAssignInstruction.getAssignToLocalVariable();

         final LocalVariableInfo localVariableInfo = assignToLocalVariable.getLocalVariableInfo();
         if (assignToLocalVariable.isDeclaration()) {
            // this is bad! we need a general way to hoist up a required declaration
            throw new CodeGenException("/* we can't declare this " + convertType(localVariableInfo.getVariableDescriptor(), true)
                  + " here */");
         }
         write(localVariableInfo.getVariableName());
         write("=");
         writeInstruction(inlineAssignInstruction.getRhs());
      } else if (_instruction.getByteCode().equals(ByteCode.FIELD_ARRAY_ELEMENT_ASSIGN)) {
         final FieldArrayElementAssign inlineAssignInstruction = (FieldArrayElementAssign) _instruction;
         final AssignToArrayElement arrayAssignmentInstruction = inlineAssignInstruction.getAssignToArrayElement();

         writeInstruction(arrayAssignmentInstruction.getArrayRef());
         write("[");
         writeInstruction(arrayAssignmentInstruction.getArrayIndex());
         write("]");
         write(" ");
         write(" = ");

         writeInstruction(inlineAssignInstruction.getRhs());
      } else if (_instruction.getByteCode().equals(ByteCode.FIELD_ARRAY_ELEMENT_INCREMENT)) {

         final FieldArrayElementIncrement fieldArrayElementIncrement = (FieldArrayElementIncrement) _instruction;
         final AssignToArrayElement arrayAssignmentInstruction = fieldArrayElementIncrement.getAssignToArrayElement();
         if (fieldArrayElementIncrement.isPre()) {
            if (fieldArrayElementIncrement.isInc()) {
               write("++");
            } else {
               write("--");
            }
         }
         writeInstruction(arrayAssignmentInstruction.getArrayRef());

         write("[");
         writeInstruction(arrayAssignmentInstruction.getArrayIndex());
         write("]");
         if (!fieldArrayElementIncrement.isPre()) {
            if (fieldArrayElementIncrement.isInc()) {
               write("++");
            } else {
               write("--");
            }
         }

      } else if (_instruction.getByteCode().equals(ByteCode.NONE)) {
         // we are done
      } else if (_instruction instanceof Branch) {
         throw new CodeGenException(String.format("%s -> %04d", _instruction.getByteCode().toString().toLowerCase(),
               ((Branch) _instruction).getTarget().getThisPC()));
      } else if (_instruction instanceof I_POP) {
         //POP discarded void call return?
         writeInstruction(_instruction.getFirstChild());
      } else {
         throw new CodeGenException(String.format("%s", _instruction.getByteCode().toString().toLowerCase()));
      }

   }

   public void writeMethod(MethodCall _methodCall, MethodEntry _methodEntry) throws CodeGenException {
      if (_methodCall instanceof VirtualMethodCall) {
         final Instruction instanceInstruction = ((VirtualMethodCall) _methodCall).getInstanceReference();
         if (!(instanceInstruction instanceof I_ALOAD_0)) {
            writeInstruction(instanceInstruction);
            write(".");
         } else {
            writeThisRef();
         }
      }
      final int argc = _methodEntry.getStackConsumeCount();
      write(_methodEntry.getNameAndTypeEntry().getNameUTF8Entry().getUTF8());
      write("(");

      for (int arg = 0; arg < argc; arg++) {
         if (arg != 0) {
            write(", ");
         }
         Instruction argObj = _methodCall.getArg(arg);
         writeInstruction(argObj);
      }
      write(")");
   }

   public void writeThisRef() {
      write("this.");
   }

   public void writeMethodBody(MethodModel _methodModel) throws CodeGenException {
      this.currentMethodBody = _methodModel.getName();
      writeBlock(_methodModel.getExprHead(), null);
      this.currentMethodBody = null;
   }

   public abstract void write(Entrypoint entryPoint) throws CodeGenException;

   public static class LocalVar {
       private final String methodName;
       private String varName;
       private int argOffset;

       public LocalVar(String setMethod, String setVar) {
           this.methodName = setMethod;
           this.varName = setVar;
           this.argOffset = -1;
       }

       public LocalVar(String setMethod, int setArg) {
           this.methodName = setMethod;
           this.argOffset = setArg;
           this.varName = null;
       }

       public String methodName() { return this.methodName; }
       public String varName() { return this.varName; }
       public int argOffset() { return this.argOffset; }
       public void resolve(HashMap<String, MethodArgumentList> methodsToArgs) {
           if (argOffset == -1) return;
           MethodArgumentList args = methodsToArgs.get(methodName);
           if (args == null) {
               return;
               // throw new RuntimeException("Missing argument info for "+methodName);
           }
           String argName = args.getArg(this.argOffset);
           this.varName = argName;
           this.argOffset = -1;
       }

       public boolean isMapLocal() {
           final String mapSuffix = "__map";
           return this.methodName.indexOf(mapSuffix) ==
                   this.methodName.length() - mapSuffix.length();
       }

       public boolean isMapParameter(HashMap<String, MethodArgumentList> methodArgs) {
           if (this.isMapLocal()) {
               MethodArgumentList allArgs = methodArgs.get(this.methodName);
               return allArgs.contains(this.varName);
           } else {
               return false;
           }
       }

       @Override
       public int hashCode() {
           return this.methodName.hashCode();
       }
       @Override
       public boolean equals(Object obj) {
           if (obj instanceof LocalVar) {
               LocalVar other = (LocalVar)obj;
               if (this.methodName.equals(other.methodName)) {
                   if (this.varName == null) {
                       return this.argOffset == other.argOffset;
                   } else {
                       return other.varName != null &&
                           this.varName.equals(other.varName);
                   }
               }
           }
           return false;
       }
       @Override
       public String toString() {
           return "["+this.methodName+":"+
               (this.varName == null ? this.argOffset : this.varName)+"]";
       }
   }

   public static class VarAlias {
       private final LocalVar passedToFunc;
       private final LocalVar funcArg;

       public VarAlias(LocalVar passed, LocalVar arg) {
           this.passedToFunc = passed;
           this.funcArg = arg;
       }

       public LocalVar getBeingPassed() {
           return passedToFunc;
       }

       public LocalVar getBeingPassedAs() {
           return this.funcArg;
       }

       @Override
       public int hashCode() {
           return this.passedToFunc.hashCode();
       }
       @Override
       public boolean equals(Object obj) {
           if (obj instanceof VarAlias) {
               VarAlias other = (VarAlias)obj;
               return other.passedToFunc.equals(this.passedToFunc) &&
                   other.funcArg.equals(this.funcArg);
           }
           return false;
       }
       @Override
       public String toString() {
           return "{ "+passedToFunc.toString()+" -> "+funcArg.toString()+" }";
       }
   }

   public static class MethodArgumentList {
       private final String methodName;
       private final List<String> methodArguments;

       public MethodArgumentList(String methodName, List<String> setMethodArguments) {
           this.methodName = methodName;
           this.methodArguments = setMethodArguments;
       }

       public String getArg(int index) {
           return this.methodArguments.get(index);
       }

       public int size() { 
           return this.methodArguments.size();
       }

       public boolean contains(String argName) {
           return this.methodArguments.contains(argName);
       }

       @Override
       public String toString() {
           StringBuffer sb = new StringBuffer();
           sb.append(this.methodName);
           sb.append("(");
           boolean first = true;
           for(String arg : this.methodArguments) {
               if (!first) sb.append(",");
               sb.append(" ");
               sb.append(arg);
               first = false;
           }
           sb.append(")");
           return sb.toString();
       }
   }
}
