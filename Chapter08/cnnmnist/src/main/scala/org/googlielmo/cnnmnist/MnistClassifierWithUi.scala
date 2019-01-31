package org.googlielmo.cnnmnist

import java.io.File
import java.util.Random

import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.datavec.api.util.ClassPathResource
import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.api.OptimizationAlgorithm

object MnistClassifierWithUi {
  @throws[Exception]
  def main(args: Array[String]): Unit = {
    val height = 28
    val width = 28
    val channels = 1 
    val outputNum = 10 
    val batchSize = 54
    val nEpochs = 1
    val iterations = 1
    
    val seed = 1234
    val randNumGen = new Random(seed)
    
    println("Data load and vectorization...")
    // Vectorization of train data
    val trainData = new ClassPathResource("/mnist_png/training").getFile
    val trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
    val labelMaker = new ParentPathLabelGenerator(); 
    val trainRR = new ImageRecordReader(height, width, channels, labelMaker);
    trainRR.initialize(trainSplit);
    val trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);
    
    // Pixel values from 0-255 to 0-1 (min-max scaling)
    val scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(trainIter);
    trainIter.setPreProcessor(scaler);
    
    // Network configuration
    val conf = new NeuralNetConfiguration.Builder()
    .seed(seed)
    .l2(0.0005)
    .weightInit(WeightInit.XAVIER)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .updater(Updater.NESTEROVS)
    .list
    .layer(0, new ConvolutionLayer.Builder(5, 5)
      .nIn(channels)
      .stride(1, 1)
      .nOut(20)
      .activation(Activation.IDENTITY)
      .build)
    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(2, 2)
      .stride(2, 2)
      .build)
    .layer(2, new ConvolutionLayer.Builder(5, 5)
      .stride(1, 1)
      .nOut(50)
      .activation(Activation.IDENTITY)
      .build)
    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(2, 2)
      .stride(2, 2)
      .build)
    .layer(4, new DenseLayer.Builder()
      .activation(Activation.RELU)
      .nOut(500)
      .build)
    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
      .nOut(outputNum)
      .activation(Activation.SOFTMAX).build)
    .setInputType(InputType.convolutionalFlat(28, 28, 1)) 
    .backprop(true).pretrain(false).build
    
    // Init the model
    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(10))
    
    // Initialize the UI
    val uiServer = UIServer.getInstance();
    
    val statsStorage = new InMemoryStatsStorage();
    
    val listenerFrequency = 1;
    model.setListeners(new StatsListener(statsStorage, listenerFrequency));
    
    uiServer.attach(statsStorage);
    
    // Training
    for (i <- 0 until nEpochs) {
      model.fit(trainIter)
      println("Completed epoch {}", i)
      trainIter.reset();
    }
    
    // Save the serialized model
    ModelSerializer.writeModel(model, new File(System.getProperty("user.home") + "/minist-model.zip"), true);
  }
}