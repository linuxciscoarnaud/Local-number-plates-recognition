/**
 * 
 */
package com.licenseplaterecogniton;

import java.io.File;
import java.util.List;
import java.util.Random;

import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import com.licenseplaterecogniton.impl.NprLabelProvider;
import com.licenseplaterecogniton.utils.LabelMatch;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * @author Arnaud
 *
 */
public class LicensePlateRecogniton {

	// parameters matching the pretrained TinyYOLO model
    int width = 416;
    int height = 416;
    int nChannels = 3;
    int gridWidth = 13;
    int gridHeight = 13;
    
    // number of classes (digits + letters) for the local license plates
    int nClasses = 36;
    
    // parameters for the Yolo2OutputLayer
    int nBoxes = 10;
    double lambdaNoObj = 0.5;
    double lambdaCoord = 1.0;
    double[][] priorBoxes = {{2, 5}, {2.5, 6}, {3, 7}, {3.5, 8}, {4, 9}, {4.5, 10}, {5, 11}, {5.5, 12}, {6, 13}, {6.5, 14}};
    double detectionThreshold = 0.5;
    
    // parameters for the training phase
    int batchSize = 10;
    int nEpochs = 10;
    double learningRate = 1e-4;
    double lrMomentum = 0.9;

    int seed = 123;
    Random rng = new Random(seed);
    
    // Variables to store start/stop training time
    private long startTrainTime = 0;
    private long endTrainTime = 0;
    
    // Variables to store start/stop training time for each epoch
    private long startEpochTime = 0;
    private long endEpochTime = 0;
	
	public void execute(String[] args) throws Exception {
		
		/**
         * Loading the data
        **/
		
		System.out.println("Loading data....");
		File trainDir = new File(System.getProperty("user.dir"), "/src/main/resources/trainData/"); 
        File testDir = new File(System.getProperty("user.dir"), "/src/main/resources/testData/");
        
        // Split up the root directory of train images in to files
        FileSplit trainData = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng);
        // Split up the root directory of test images in to files
        FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);
        
        // Image record reader for object detection. 
        // The format of returned values: 4d array, with dimensions [minibatch, 4+C, h, w], where the image is quantized into h x w grid locations.
        // so as to match the format required for Deeplearning4j's Yolo2OutputLayer
        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new NprLabelProvider(trainDir));
        recordReaderTrain.initialize(trainData);
        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new NprLabelProvider(testDir));
        recordReaderTest.initialize(testData);
        
        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        
        /**
         * Building model...
        **/
        
        System.out.println("Building model...");
        ComputationGraph model;
        String modelFilename = "model.zip";
        
        if (new File(modelFilename).exists()) {
        	System.out.println("Load model...");
        	
        	model = ModelSerializer.restoreComputationGraph(modelFilename);
        }else {
        	System.out.println("Build model...");
        	
        	ComputationGraph pretrained = (ComputationGraph)TinyYOLO.builder().build().initPretrained();
        	
        	// It's important for me that i save the weights this pretrained model
        	//ModelSerializer.writeModel(pretrained, "tiny-yolo-voc_dl4j_inference.v1.zip", false);
        	
        	System.out.println(pretrained.summary(InputType.convolutional(height, width, nChannels)));
        	
        	INDArray priors = Nd4j.create(priorBoxes);
        	
        	// Configuration for fine tuning. Note that values set here will override values for all non-frozen layers
            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            		.seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    .updater(new Adam.Builder().learningRate(learningRate).build())
                    //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                    .l2(0.00001)
                    .activation(Activation.IDENTITY)
                    .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                    .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
            		.build();
            
            model = new TransferLearning.GraphBuilder(pretrained)
            		.fineTuneConfiguration(fineTuneConf)
            		.removeVertexKeepConnections("conv2d_9")
                    .removeVertexKeepConnections("outputs")
                    .addLayer("convolution2d_9",
                    		new ConvolutionLayer.Builder(1,1)
                    		        .nIn(1024)
                                    .nOut(nBoxes * (5 + nClasses))
                                    .stride(1,1)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(Activation.IDENTITY)
                    		        .build(),
                    		"leaky_re_lu_8")
                    .addLayer("outputs",
                    		new Yolo2OutputLayer.Builder()
                    		        .lambbaNoObj(lambdaNoObj)
                                    .lambdaCoord(lambdaCoord)
                                    .boundingBoxPriors(priors)
                                    .build(),
                    		"convolution2d_9")
                    .setOutputs("outputs")
            		.build();
            System.out.println(model.summary(InputType.convolutional(height, width, nChannels)));
            
            /**
             * Training model...
            **/
            
            System.out.println("Train model...");
            model.setListeners(new ScoreIterationListener(1));
            startTrainTime = System.currentTimeMillis();
            for (int i = 0; i < nEpochs; i++) {
            	train.reset();
            	startEpochTime = System.currentTimeMillis();
            	while (train.hasNext()) {
            		model.fit(train.next());
            	}
            	endEpochTime = System.currentTimeMillis();
            	System.out.println("*** Epoch " + i + " completed in " + (endEpochTime - startEpochTime) / 60000.0 + " min");
            }
            endTrainTime = System.currentTimeMillis();
            System.out.println("****************End of Training********************");           
            System.out.println("Training time...: " + (endTrainTime - startTrainTime) / 60000.0 + " min");
            System.out.println();
            
            System.out.println("Saving model...");
            ModelSerializer.writeModel(model, modelFilename, true);
            System.out.println();
        }
        
        /**
         * Evaluating model...
        **/
        
        System.out.println("Evaluating model...");
        // This will use JavaCV to load images. 
        // Allowed formats: bmp, gif, jpg, jpeg, jp2, pbm, pgm, ppm, pnm, png, tif, tiff, exr, webp
        NativeImageLoader imageLoader = new NativeImageLoader();
        
        LabelMatch labelMatch = new LabelMatch();
        
        // Output (loss) layer for YOLOv2 object detection model
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout =
                (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer)model.getOutputLayer(0); // 0 because total number of network outputs = 1
        
        // Get the list of labels
        List<String> labels = train.getLabels();
        
        // Make sure metadata for the current examples is present in the returned DataSet
        test.setCollectMetaData(true);
        
        while (test.hasNext()) {
        	org.nd4j.linalg.dataset.DataSet ds = test.next();       	    
            RecordMetaDataImageURI metadata = (RecordMetaDataImageURI)ds.getExampleMetaData().get(0);                 
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            System.out.print(objs.size()+ " objects detected in file: ");
            File file = new File(metadata.getURI());
            //System.out.println(file.getName() + ": " + objs);
            System.out.print(file.getName() + ": ");
            for (DetectedObject obj : objs) {
            	System.out.print(labelMatch.matchLetterOrDigits(obj.getPredictedClass())+ "  ");
            }
            System.out.println();
        }
	}
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		new LicensePlateRecogniton().execute(args);
	}
}
