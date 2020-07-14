package com.mardillu.audiverify;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;

import com.google.firebase.ml.custom.FirebaseCustomLocalModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelInterpreterOptions;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity1";

    Classifier classifier;
    Device device = Device.CPU;
    /** An Interpreter for the TFLite model.   */
    Interpreter interpreter;
    private GpuDelegate gpuDelegate;
    private int NUM_LITE_THREADS = 4;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

    }

    /**
     * LOAD RAW KERAS MODEL USING deeplearning4j LIBRARY
     * First uncomment implementation('org.deeplearning4j:deeplearning4j-modelimport:1.0.0-beta7') in build.gradle
     *
     * Anyways, the project won't build
     *
     * @param v
     */
    public void loadModelWithDeepLearning4J(View v){
//        try {
//            InputStream inputStream = getAssets().open("trainedh5.h5");
//            MultiLayerNetwork network = KerasModelImport.importKerasSequentialModelAndWeights(inputStream);
//        } catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
//            e.printStackTrace();
//        }
    }

    /**
     * LOAD tflite MODEL FirebaseCustomLocalModel
     *
     * Firebase loads the model correctly.
     * How to get a 4D float32 array from a wav file to feed into model??
     *
     * @param v
     */
    public void loadModelWithFirbaseML(View v){
        try {
            //load the model
            FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder()
                    .setAssetFilePath("convertedh5_model.tflite")
                    .build();
            FirebaseModelInterpreterOptions options =
                    new FirebaseModelInterpreterOptions.Builder(localModel).build();
            FirebaseModelInterpreter interp = FirebaseModelInterpreter.getInstance(options);

            //define input and output options
//            import tensorflow as tf
//            interpreter = tf.lite.Interpreter(model_path="models/converted_model_default_quant.tflite")
//            interpreter.allocate_tensors()
//            # Print input shape and type
//            print(interpreter.get_input_details()[0]['shape'])  # Example: [1 160 64 1]
//            print(interpreter.get_input_details()[0]['dtype'])  # Example: <class 'numpy.float32'>
//            # Print output shape and type
//            print(interpreter.get_output_details()[0]['shape'])  # Example: [1 512]
//            print(interpreter.get_output_details()[0]['dtype'])

            FirebaseModelInputOutputOptions inputOutputOptions =
                    new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 160, 64, 1})
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 512})
                            .build();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    /** LOAD tflite MODEL using teensorflow
     *
     * Loading model app crashes with bytebuffer error
     *
     * @param v
     */
    public void loadModelWithTensorFlow(View v){
        try {
            interpreter = getInterpreter();
            Log.d(TAG, "onCreate: " + interpreter.getLastNativeInferenceDurationNanoseconds());
            Log.d(TAG, "onCreate: " + interpreter.getOutputTensorCount());
            Log.d(TAG, "onCreate: " + interpreter.getInputTensorCount());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Load a pytorch model with Pytorch
     *
     * Loading model crashes app with invalid model error
     *
     * @param v
     */
    public void loadModelWithPytorch(View v){
        try {
            classifier = new Classifier(Utils.assetFilePath(this,"traced_model.pt"));
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    /**
     * Get a TensorFlow interpreter
     * @return
     * @throws IOException
     */
    private Interpreter getInterpreter() throws IOException {
        if (interpreter != null) {
            return interpreter;
        }
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(NUM_LITE_THREADS);
        switch (device) {
            case CPU:
                break;
            case GPU:
                gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
            break;
            case NNAPI:
                options.setUseNNAPI(true);
                break;
        }
        interpreter = new Interpreter(loadModelFile3(), options);
        return interpreter;
    }

    /** Preload and memory map the model file, returning a MappedByteBuffer containing the model. */
    private File loadModelFile2() throws IOException {
        InputStream inputStream = getAssets().open("convertedh5_model.tflite");
       return createFileFromInputStream(inputStream);
    }


    /** Preload and memory map the model file, returning a MappedByteBuffer containing the model. */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("convertedh5_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        return inputStream.getChannel().map(FileChannel.MapMode.READ_ONLY, fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile3() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("converted_model_default_quant.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private File createFileFromInputStream(InputStream inputStream) {
        try{
            File f = new File(getCacheDir()+"/modl.tflite");
            OutputStream outputStream = new FileOutputStream(f);
            byte buffer[] = new byte[1024];
            int length = 0;

            while((length=inputStream.read(buffer)) > 0) {
                outputStream.write(buffer,0,length);
            }
            outputStream.close();
            inputStream.close();

            return f;
        }catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }



    enum Device {CPU, NNAPI, GPU }
}
