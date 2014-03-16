import java.io.*;

import org.apache.hadoop.mapreduce.HadoopOpenCLContext;
import java.lang.reflect.Constructor;
import java.net.MalformedURLException;
import com.amd.aparapi.*;
import com.amd.aparapi.device.*;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;
import com.amd.aparapi.internal.util.*;
import com.amd.aparapi.internal.opencl.*;

public class OpenCLGenerator {

    public static void main(String[] args) throws MalformedURLException {
        if (args.length != 3) {
            System.out.println("usage: java OpenCLGenerator <class-name> <strided> <exec-mode>");
            return;
        }
        Kernel.EXECUTION_MODE exec;
        String className = args[0];
        boolean strided;

        if (args[1].equals("true") || args[1].equals("t")) {
            strided = true;
        } else if (args[1].equals("false") || args[1].equals("f")) {
            strided = false;
        } else {
            System.out.println("Invalid value \""+args[1]+"\" specified for strided, must be true/t or false/f");
            return;
        }

        final OpenCLDevice device;
        if (args[2].equals("gpu") || args[2].equals("g")) {
            exec = Kernel.EXECUTION_MODE.GPU;
            device = findDevice(findDeviceWithType(Device.TYPE.GPU));
        } else if (args[2].equals("cpu") || args[2].equals("c")) {
            exec = Kernel.EXECUTION_MODE.CPU;
            device = findDevice(findDeviceWithType(Device.TYPE.CPU));
        } else {
            System.out.println("Invalid value \""+args[2]+"\" specified for exec mode, must be cpu/c or gpu/g");
            return;
        }

        try {

          Class c = Class.forName(className);
          Constructor<? extends Kernel> construct = c.getConstructor(new Class[] { HadoopOpenCLContext.class, Integer.class });
          Kernel a = construct.newInstance(new HadoopOpenCLContext(), -1);
            a.setStrided(strided);
            a.setExecutionMode(exec);
            a.execute(device.createRange(128), 0, true, 0, 0, null);
            // a.execute(device.createRange(128), 0, "foo"); // dryRun = true
        } catch (Exception e) {
            System.out.println(e);
            e.printStackTrace();
        }

    }

    private static int findDeviceWithType(Device.TYPE type) {
        int devicesSoFar = 0;
        List<OpenCLPlatform> platforms = OpenCLUtil.getOpenCLPlatforms();
        for(OpenCLPlatform platform : platforms) {
            for(OpenCLDevice tmpDev : platform.getOpenCLDevices()) {
              if(tmpDev.getType() == type) {
                return devicesSoFar;
              }
              devicesSoFar++;
            }
        }
        return -1;
    }

    public static OpenCLDevice findDevice(int id) {
        int devicesSoFar = 0;
        OpenCLDevice dev = null;
        List<OpenCLPlatform> platforms = OpenCLUtil.getOpenCLPlatforms();
        for(OpenCLPlatform platform : platforms) {
            for(OpenCLDevice tmpDev : platform.getOpenCLDevices()) {
              if(devicesSoFar == id) {
                dev = tmpDev;
                break;
              }
              devicesSoFar++;
            }
            if(dev != null) break;
        }
        return dev;
    }
}
