package org.googlielmo.cifar10scala

import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.CacheMode
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.DropoutLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.conf.layers.PoolingType
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext

import scala.collection.mutable

object Cifar10SimpleSpark {
  private val channels = 3
  private val height = 32
	private val width = 32
	private val numLabels = CifarLoader.NUM_LABELS
	private val seed = 123
	private val epochs = 1
	private val batchSize = 10
	
	@throws[Exception]
   def main(args: Array[String]) {
    val trainDataSetIterator = 
                new CifarDataSetIterator(2, 5000, true)
     val testDataSetIterator = 
                new CifarDataSetIterator(2, 200, false)
     println(trainDataSetIterator.getLabels)
     
     // Create the neural network
		 val conf = defineModelConfiguration
		 val model = new MultiLayerNetwork(conf)
     model.init
     
     // Init the Spark context
     val sparkConf = new SparkConf
     sparkConf.setMaster("local[*]")
      .setAppName("DL4J Spark Cifar-10 Example")
     val sc = new JavaSparkContext(sparkConf)
     
     // Parallelize data
     val trainDataList = mutable.ArrayBuffer.empty[DataSet]
     while (trainDataSetIterator.hasNext) {
      trainDataList += trainDataSetIterator.next
     }
     val paralleltrainData = sc.parallelize(trainDataList)
     
     // Create the TrainingMaster
     var batchSizePerWorker: Int = 16
     val tm = new
      ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)
      .batchSizePerWorker(batchSizePerWorker)
      .build
     
     // Training
     val sparkNet = new SparkDl4jMultiLayer(sc, conf, tm)
     for (i <- 0 until epochs) {
      sparkNet.fit(paralleltrainData)
      println("Completed Epoch {}", i)
     }
     
     sc.stop()
     println("--- Application end. ---")
  }
  
  def defineModelConfiguration(): MultiLayerConfiguration =
     new NeuralNetConfiguration.Builder()
            .seed(seed)
            .cacheMode(CacheMode.DEVICE)
            .updater(new Adam(1e-2))
            .biasUpdater(new Adam(1e-2*2))
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) 
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .l1(1e-4)
            .l2(5 * 1e-4)
            .list
            .layer(0, new ConvolutionLayer.Builder(Array(4, 4), Array(1, 1), Array(0, 0)).name("cnn1").convolutionMode(ConvolutionMode.Same)
                .nIn(3).nOut(64).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .biasInit(1e-2).build)
            .layer(1, new ConvolutionLayer.Builder(Array(4, 4), Array(1, 1), Array(0, 0)).name("cnn2").convolutionMode(ConvolutionMode.Same)
                .nOut(64).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .biasInit(1e-2).build)
            .layer(2, new SubsamplingLayer.Builder(PoolingType.MAX, Array(2,2)).name("maxpool2").build())

            .layer(3, new ConvolutionLayer.Builder(Array(4, 4), Array(1, 1), Array(0, 0)).name("cnn3").convolutionMode(ConvolutionMode.Same)
                .nOut(96).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .biasInit(1e-2).build)
            .layer(4, new ConvolutionLayer.Builder(Array(4, 4), Array(1, 1), Array(0, 0)).name("cnn4").convolutionMode(ConvolutionMode.Same)
                .nOut(96).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .biasInit(1e-2).build)

            .layer(5, new ConvolutionLayer.Builder(Array(3,3), Array(1, 1), Array(0, 0)).name("cnn5").convolutionMode(ConvolutionMode.Same)
                .nOut(128).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .biasInit(1e-2).build)
            .layer(6, new ConvolutionLayer.Builder(Array(3,3), Array(1, 1), Array(0, 0)).name("cnn6").convolutionMode(ConvolutionMode.Same)
                .nOut(128).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .biasInit(1e-2).build)

            .layer(7, new ConvolutionLayer.Builder(Array(2,2), Array(1, 1), Array(0, 0)).name("cnn7").convolutionMode(ConvolutionMode.Same)
                .nOut(256).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .biasInit(1e-2).build)
            .layer(8, new ConvolutionLayer.Builder(Array(2,2), Array(1, 1), Array(0, 0)).name("cnn8").convolutionMode(ConvolutionMode.Same)
                .nOut(256).weightInit(WeightInit.XAVIER_UNIFORM).activation(Activation.RELU)
                .biasInit(1e-2).build)
            .layer(9, new SubsamplingLayer.Builder(PoolingType.MAX, Array(2,2)).name("maxpool8").build())

            .layer(10, new DenseLayer.Builder().name("ffn1").nOut(1024).updater(new Adam(1e-3)).biasInit(1e-3).biasUpdater(new Adam(1e-3*2)).build)
            .layer(11,new DropoutLayer.Builder().name("dropout1").dropOut(0.2).build)
            .layer(12, new DenseLayer.Builder().name("ffn2").nOut(1024).biasInit(1e-2).build)
            .layer(13,new DropoutLayer.Builder().name("dropout2").dropOut(0.2).build)
            .layer(14, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(numLabels)
                .activation(Activation.SOFTMAX)
                .build)
            .backprop(true)
            .pretrain(false)
            .setInputType(InputType.convolutional(height, width, channels))
            .build
}
