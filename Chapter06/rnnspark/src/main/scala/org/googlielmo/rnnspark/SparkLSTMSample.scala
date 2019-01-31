package org.googlielmo.rnnspark

import java.nio.file.Files
import java.nio.charset.Charset
import java.io.{File, IOException}
import java.util.{ArrayList, Collections, Random}
import org.apache.spark.SparkConf
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.broadcast.Broadcast
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{BackpropType, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import scala.collection.mutable

object SparkLSTMSample {
  private val CHAR_TO_INT: Map[Char, Int] = SparkLSTMSample.getCharToInt
  private val INT_TO_CHAR: Map[Int, Char] = SparkLSTMSample.getIntToChar
  private val N_CHARS = INT_TO_CHAR.size
  private val nOut = CHAR_TO_INT.size
  private val exampleLength = 1000 
  
  def getCharToInt: Map[Char, Int] = {
    val m = mutable.Map.empty[Char, Int]
    val chars = getValidCharacters
    chars.indices.foreach { i =>
      m.update(chars(i), i)
    }
    m.toMap
  }
  
  def getIntToChar: Map[Int, Char] = {
      val m = mutable.Map.empty[Int, Char]
      val chars = getValidCharacters
      chars.indices.foreach { i =>
        m.update(i, chars(i))
      }
      m.toMap
  }
  
  /**
    * Returns a minimal character set, with a-z, A-Z, 0-9 and common punctuation.
    */
  private def getValidCharacters: Array[Char] = {
    val validChars = mutable.ArrayBuffer.empty[Char]
    val chars = ('a' to 'z') ++
      ('A' to 'Z') ++
      ('0' to '9') ++
      Seq('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')

    chars.foreach { c =>
      validChars += c
    }
    validChars.toArray
  }
  
  /**
    * Loads data from a file and remove any invalid characters.
    */
  @throws[IOException]
  private def getDataAsString(filePath: String): String = {
    val lines = Files.readAllLines(new File(filePath).toPath, Charset.defaultCharset)
    val sb = new StringBuilder
    import scala.collection.JavaConversions._
    for (line <- lines) {
      val chars: Array[Char] = line.toCharArray
      chars.indices.foreach { i =>
        if (SparkLSTMSample.CHAR_TO_INT.containsKey(chars(i))) sb.append(chars(i))
      }
      sb.append("\n")
    }
    sb.toString
  }
  
  /**
    * Transforms the input text into a List<String>.
    */
  @throws[IOException]
  private def getInputTextAsList(sequenceLength: Int): List[String] = {
    val fileLocation = getClass().getClassLoader().getResource("pg100sub.txt").getPath
    val allData = getDataAsString(fileLocation)

    val list = mutable.ArrayBuffer.empty[String]
    val length = allData.length
    var currIdx = 0
    while (currIdx + sequenceLength < length) {
      val end = currIdx + sequenceLength
      val substr: String = allData.substring(currIdx, end)
      currIdx = end
      list += substr
    }
    list.toList
  }
  
  @throws[IOException]
  def getTrainingData(sc: JavaSparkContext): JavaRDD[DataSet] = {
    val list = getInputTextAsList(exampleLength)
    val rawStrings = sc.parallelize(list)
    val bcCharToInt = sc.broadcast(CHAR_TO_INT)
    rawStrings.map(stringToDataSetFn(bcCharToInt))
  }
  
  @throws[Exception]
  def stringToDataSetFn(ctiBroadcast: Broadcast[Map[Char, Int]])(s: String): DataSet = {
    val cti: Map[Char, Int] = ctiBroadcast.value
    val length: Int = s.length
    val features: INDArray = Nd4j.zeros(1, N_CHARS, length - 1)
    val labels: INDArray = Nd4j.zeros(1, N_CHARS, length - 1)
    val chars: Array[Char] = s.toCharArray
    val f: Array[Int] = new Array[Int](3)
    val l: Array[Int] = new Array[Int](3)
    var i: Int = 0
    (0 until chars.length - 2).foreach { i =>
      f(1) = cti(chars(i))
      f(2) = i
      l(1) = cti(chars(i + 1)) 
      l(2) = i
      features.putScalar(f, 1.0)
      labels.putScalar(l, 1.0)
    }
    new DataSet(features, labels)
  }
  
  def main(args: Array[String]) {
    val rng = new Random(12345)
    val lstmLayerSize: Int = 200 
    val tbpttLength: Int = 50 
    val nSamplesToGenerate: Int = 4 
    val nCharactersToSample: Int = 300 
    val generationInitialization: String = null
      
    // Set up the network configuration:
    val conf = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.1)
      .rmsDecay(0.95)
      .seed(12345)
      .regularization(true)
      .l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.RMSPROP)
      .list
      .layer(0, new GravesLSTM.Builder().nIn(SparkLSTMSample.CHAR_TO_INT.size).nOut(lstmLayerSize).activation(Activation.TANH).build())
      .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize).activation(Activation.TANH).build())
      .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)
        .nIn(lstmLayerSize).nOut(SparkLSTMSample.nOut).build) //MCXENT + softmax for classification
      .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
      .pretrain(false).backprop(true)
      .build
      
    //Set up the Spark configuration
    val averagingFrequency: Int = 3

    val sparkConf = new SparkConf
    sparkConf.setMaster("local[*]")
    
    sparkConf.setAppName("LSTM Character Example")
    val sc = new JavaSparkContext(sparkConf)
    sc.setLogLevel("WARN")
    
    // Get the training data
    val trainingData = getTrainingData(sc)
    
    // Set up the TrainingMaster
    val batchSizePerWorker: Int = 8
    val examplesPerDataSetObject = 1
    val tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
      .workerPrefetchNumBatches(2)
      .averagingFrequency(averagingFrequency)  
      .batchSizePerWorker(batchSizePerWorker)
      .build
    val sparkNetwork: SparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc, conf, tm)
    sparkNetwork.setListeners(Collections.singletonList[IterationListener](new ScoreIterationListener(1)))
    
    // Start traiing and then generate and print samples from the network
    val numEpochs: Int = 1
    (0 until numEpochs).foreach { i =>
      val net = sparkNetwork.fit(trainingData) 

      println("Sampling characters from network given initialization \"" +
        (if (generationInitialization == null) "" else generationInitialization) + "\"")
        val samples = sampleCharactersFromNetwork(generationInitialization, net, rng, SparkLSTMSample.INT_TO_CHAR,
          nCharactersToSample, nSamplesToGenerate)
      
      samples.indices.foreach { j =>
        println("----- Sample " + j + " -----")
        println(samples(j))
      }
      
      println("--Epoch # --" + i + " completed.")
    }

    // Delete the temp training files at the end
    tm.deleteTempFiles(sc)
    println("\n\nExample complete")
    
    // Stop the Spark context
    sc.stop()
  }
  
  /**
    * Generate a sample from the network, given an (optional, possibly null) initialization.
    */
  private def sampleCharactersFromNetwork(initialization: String, net: MultiLayerNetwork, rng: Random, intToChar: Map[Int, Char], charactersToSample: Int, numSamples: Int): Array[String] = {
    val _initialization = ""
    
    val initializationInput = Nd4j.zeros(numSamples, intToChar.size, _initialization.length)
    val init = _initialization.toCharArray
    for (i <- init.indices) {
      val idx = SparkLSTMSample.CHAR_TO_INT(init(i))
      for (j <- 0 until numSamples) {
        initializationInput.putScalar(Array(j, idx, i), 1.0f)
      }
    }
    val sb = new Array[StringBuilder](numSamples)
    for (i <- 0 until numSamples) {
      sb(i) = new StringBuilder(_initialization)
    }
    
    net.rnnClearPreviousState()
    var output = net.rnnTimeStep(initializationInput)
    output = output.tensorAlongDimension(output.size(2) - 1, 1, 0) 

    for (i <- 0 until charactersToSample) {
      val nextInput: INDArray = Nd4j.zeros(numSamples, intToChar.size)
      
      for (s <- 0 until numSamples) {
        val outputProbDistribution: Array[Double] = new Array[Double](intToChar.size)
          for (j <- outputProbDistribution.indices) {
            outputProbDistribution(j) = output.getDouble(s, j)
          }
          val sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng)
          nextInput.putScalar(Array(s, sampledCharacterIdx), 1.0f) 
          sb(s).append(intToChar.get(sampledCharacterIdx)) 
      }
      output = net.rnnTimeStep(nextInput) 
    }
    var out = new mutable.ArrayBuffer[String](numSamples)
    for (i <- 0 until numSamples) {
      out += sb(i).toString
    }
    out.toArray
  }

  private def sampleFromDistribution(distribution: Array[Double], rng: Random): Int = {
    val d = rng.nextDouble
    val i = distribution
      .toIterator
      .scanLeft(0.0)({ case (acc, p) => acc + p })
      .drop(1)
      .indexWhere(_ >= d)
    if (i >= 0) {
      i
    } else {
      throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + distribution.sum)
    }
  }
}
