package org.googlielmo.cnnscalabench

import java.io.File
import java.util.Random

import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.datavec.api.util.ClassPathResource
import org.datavec.image.loader.NativeImageLoader
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
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
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.eval.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet

import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext

import scala.collection.mutable

import org.slf4j.{Logger, LoggerFactory}

object MnistClassifierSpark {
  private val log: Logger = LoggerFactory.getLogger(MnistClassifierSpark.getClass)
  
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
      
      // Spark configuration
      val sparkConf = new SparkConf
      sparkConf.setMaster("local[*]")
        .setAppName("DL4J Spark MNIST Example")
      val sc = new JavaSparkContext(sparkConf)
      
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
      
      // Vectorization of test data
      val testData = new ClassPathResource("/mnist_png/testing").getFile
      val testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
      val testRR = new ImageRecordReader(height, width, channels, labelMaker)
      testRR.initialize(testSplit)
      val testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum)
      testIter.setPreProcessor(scaler) 
      
      // Load the data into memory and then parallelize
      val trainDataList = mutable.ArrayBuffer.empty[DataSet]
      val testDataList = mutable.ArrayBuffer.empty[DataSet]
      while (trainIter.hasNext) {
        trainDataList += trainIter.next
      }
      while (testIter.hasNext) {
        testDataList += testIter.next
      }
      
      val paralleltrainData = sc.parallelize(trainDataList)
      val parallelTestData = sc.parallelize(testDataList)
      
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
      
      // Training through Apache Spark
      var batchSizePerWorker: Int = 16
      val tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker) 
        .averagingFrequency(5)
        .workerPrefetchNumBatches(2)      
        .batchSizePerWorker(batchSizePerWorker)
        .build
  
      //Create the Spark network
      val sparkNet = new SparkDl4jMultiLayer(sc, conf, tm)
  
      //Execute training:
      var numEpochs: Int = 15
      var i: Int = 0
      for (i <- 0 until numEpochs) {
        sparkNet.fit(paralleltrainData)
        val eval = sparkNet.evaluate(parallelTestData)
        println(eval.stats)
        println("Completed Epoch {}", i)
        trainIter.reset
        testIter.reset
        
      }
      
      println("****************Example finished********************")

  }
}
