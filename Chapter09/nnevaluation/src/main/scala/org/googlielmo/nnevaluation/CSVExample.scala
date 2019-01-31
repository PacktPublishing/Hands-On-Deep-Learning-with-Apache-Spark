package org.googlielmo.nnevaluation

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.eval.RegressionEvaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, NormalizerStandardize}
import org.nd4j.linalg.dataset.{DataSet, SplitTestAndTrain}
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

object CSVExample {
  private val log: Logger = LoggerFactory.getLogger(CSVExample.getClass)

  @throws[Exception]
  def main(args: Array[String]) {

    // Get the Iris dataset using a CSVRecordReader reader
    val numLinesToSkip = 1
    val delimiter = ","
    val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReader.initialize(new FileSplit(new ClassPathResource("iris.csv").getFile))

    // A RecordReaderDataSetIterator handles conversion to DataSet objects, ready to be used in the neural network
    val labelIndex = 4 
    val numClasses = 3 
    val batchSize = 150 

    val iterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses)
    val allData: DataSet = iterator.next
    allData.shuffle()
    // Let's use 65% of data for training
    val testAndTrain: SplitTestAndTrain = allData.splitTestAndTrain(0.65) 

    val trainingData: DataSet = testAndTrain.getTrain
    val testData: DataSet = testAndTrain.getTest

    // Normalize the data. 
    val normalizer: DataNormalization = new NormalizerStandardize
    normalizer.fit(trainingData) 
    normalizer.transform(trainingData) 
    normalizer.transform(testData) 

    val numInputs = 4
    val outputNum = 3
    val iterationCount = 1000
    val seed = 6


    println("Build model....")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .activation(Activation.TANH)
      .weightInit(WeightInit.XAVIER)
      .l2(1e-4)
      .list
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
        .build)
      .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
        .build)
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .nIn(3).nOut(outputNum).build)
      .backprop(true).pretrain(false)
      .build

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100))

    // Start training
    for(idx <- 0 to 2000) {
      model.fit(trainingData)
    }

    //evaluate the model on the test set
//    val eval = new Evaluation(3)
//    val output = model.output(testData.getFeatureMatrix)
//    eval.eval(testData.getLabels, output)
//    // println(eval.stats)
//    println(eval.getConfusionMatrix.toHTML)
    
    // Evaluation for regression
    val eval = new RegressionEvaluation(3)
    val output = model.output(testData.getFeatureMatrix)
    eval.eval(testData.getLabels, output)
    println(eval.stats)
  }
}
