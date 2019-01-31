package org.googlielmo.rnnbase

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import java.util
import java.util.Random

object BasicRnn {
  // Sentence to learn
  val LEARNSTRING: Array[Char] = 
    "*Over the past year, there have been a number of trends that have impacted digital transformation, including artificial intelligence, big data and robotic process automation, among many others."
    .toCharArray
  
  // The list of all the possible characters
  val LEARNSTRING_CHARS_LIST: util.List[Character] = new util.ArrayList[Character]
  
  // Dimensions of the RNN
  val HIDDEN_LAYER_WIDTH = 50
  val HIDDEN_LAYER_CONT = 2
  val r = new Random(7894)
  
  def main(args: Array[String]) {
    // Create a dedicated list of possible chars in LEARNSTRING_CHARS_LIST
    val LEARNSTRING_CHARS: util.LinkedHashSet[Character] = new util.LinkedHashSet[Character]
    for (c <- LEARNSTRING) LEARNSTRING_CHARS.add(c)
    LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS)
    
    // Configure the neural network
    val builder: NeuralNetConfiguration.Builder = new NeuralNetConfiguration.Builder
    builder.iterations(10)
    builder.learningRate(0.001)
    builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    builder.seed(1234)
    builder.biasInit(0)
    builder.miniBatch(false)
    builder.updater(Updater.RMSPROP)
    builder.weightInit(WeightInit.XAVIER)
    
    val listBuilder = builder.list
    
    // Use the GravesLSTM.Builder
    for (i <- 0 until HIDDEN_LAYER_CONT) {
      val hiddenLayerBuilder: GravesLSTM.Builder = new GravesLSTM.Builder
      hiddenLayerBuilder.nIn(if (i == 0) LEARNSTRING_CHARS.size else HIDDEN_LAYER_WIDTH)
      hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH)

      hiddenLayerBuilder.activation(Activation.TANH)
      listBuilder.layer(i, hiddenLayerBuilder.build)
    }
    
    // Use RnnOutputLayer
    val outputLayerBuilder: RnnOutputLayer.Builder = new RnnOutputLayer.Builder(LossFunction.MCXENT)
    outputLayerBuilder.activation(Activation.SOFTMAX)
    outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH)
    outputLayerBuilder.nOut(LEARNSTRING_CHARS.size)
    listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build)

    listBuilder.pretrain(false)
    listBuilder.backprop(true)
    
    // Build the neural network
    val conf = listBuilder.build
    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))
    
    // Create the training data
    val input = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size, LEARNSTRING.length)
    val labels = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size, LEARNSTRING.length)
    var samplePos = 0
    for (currentChar <- LEARNSTRING) {
      val nextChar = LEARNSTRING((samplePos + 1) % (LEARNSTRING.length))
      input.putScalar(Array[Int](0, LEARNSTRING_CHARS_LIST.indexOf(currentChar), samplePos), 1)
      labels.putScalar(Array[Int](0, LEARNSTRING_CHARS_LIST.indexOf(nextChar), samplePos), 1)
      samplePos += 1
    }
    val trainingData: DataSet = new DataSet(input, labels)
    print(trainingData)
    
    // Train the model
    for (epoch <- 0 until 130) {
      println("Epoch " + epoch)

      net.fit(trainingData)

      net.rnnClearPreviousState()

      val testInit: INDArray = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size)
      testInit.putScalar(LEARNSTRING_CHARS_LIST.indexOf(LEARNSTRING(0)), 1)

      var output: INDArray = net.rnnTimeStep(testInit)

      for (j <- LEARNSTRING.indices) {
        val outputProbDistribution: Array[Double] = new Array[Double](LEARNSTRING_CHARS.size)
        for (k <- outputProbDistribution.indices) {
          outputProbDistribution(k) = output.getDouble(k)
        }
        val sampledCharacterIdx = findIndexOfHighestValue(outputProbDistribution)

        print(LEARNSTRING_CHARS_LIST.get(sampledCharacterIdx))

        val nextInput: INDArray = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size)
        nextInput.putScalar(sampledCharacterIdx, 1)
        output = net.rnnTimeStep(nextInput)
      }
      print("\n")
    }
  }
  
  private def findIndexOfHighestValue(distribution: Array[Double]): Int = {
    var maxValueIndex: Int = 0
    var maxValue: Double = 0
    for (i <- distribution.indices) {
      if (distribution(i) > maxValue) {
        maxValue = distribution(i)
        maxValueIndex = i
      }
    }
    maxValueIndex
  }
}