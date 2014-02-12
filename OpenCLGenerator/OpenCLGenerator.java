import java.io.*;

import org.apache.hadoop.mapreduce.HadoopOpenCLContext;
import java.lang.reflect.Constructor;
import java.net.MalformedURLException;
import com.amd.aparapi.*;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public class OpenCLGenerator {

    public static void main(String[] args) throws MalformedURLException {
        if (args.length != 3) {
            System.out.println("usage: java OpenCLGenerator <class-file> <strided> <exec-mode>");
            return;
        }
        Kernel.EXECUTION_MODE exec;
        String fileName = args[0];
        boolean strided;

        if (args[1].equals("true") || args[1].equals("t")) {
            strided = true;
        } else if (args[1].equals("false") || args[1].equals("f")) {
            strided = false;
        } else {
            System.out.println("Invalid value \""+args[1]+"\" specified for strided, must be true/t or false/f");
            return;
        }

        if (args[2].equals("gpu") || args[2].equals("g")) {
            exec = Kernel.EXECUTION_MODE.GPU;
        } else if (args[2].equals("cpu") || args[2].equals("c")) {
            exec = Kernel.EXECUTION_MODE.CPU;
        } else {
            System.out.println("Invalid value \""+args[2]+"\" specified for exec mode, must be cpu/c or gpu/g");
            return;
        }

        // System.out.println("class-file="+fileName+", strided="+strided+", exec="+(exec==Kernel.EXECUTION_MODE.CPU ? "cpu" : "gpu"));

        java.net.URLClassLoader loader = new java.net.URLClassLoader(new java.net.URL[] { new java.net.URL("file:///home/yiskylee/OpenCLGenerator/"+fileName) } );
        try {

          Class c = loader.loadClass(fileName.substring(0, fileName.indexOf('.')));
          Constructor<? extends Kernel> construct = c.getConstructor(new Class[] { HadoopOpenCLContext.class, Integer.class });
          Kernel a = construct.newInstance(new HadoopOpenCLContext(), -1);
            // File f = new File(fileName);
            // Class c = loader.loadClassFile(f);
            // Kernel a = (Kernel)c.newInstance();
            a.setStrided(strided);
            a.setExecutionMode(exec);
            a.execute(128, true, 0, 0); // dryRun = true
        } catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }

    }
}
