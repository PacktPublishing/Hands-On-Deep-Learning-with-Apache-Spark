package org.googlielmo.vgg16

import org.datavec.api.util.ClassPathResource
import org.datavec.image.loader.NativeImageLoader
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor

import java.io.File

object KerasImporter {
  @throws[Exception]
   def main(args: Array[String]) {
    // Load the VGG-16 pre-trained model
    val vgg16Json = new ClassPathResource("vgg-16.json").getFile.getPath
    val vgg16 = new ClassPathResource("vgg-16.h5").getFile.getPath
    val model = KerasModelImport.importKerasModelAndWeights(vgg16Json, vgg16, false)
    
    // Load the test image
    val testImage = new ClassPathResource("test_image-02.jpg").getFile
    
    val height = 224
    val width = 224
    val channels = 3
    val loader = new NativeImageLoader(height, width, channels)
    
    // Transform the image in a INDArray
    val image = loader.asMatrix(testImage)
    val scaler = new VGG16ImagePreProcessor
    scaler.transform(image)
    
    val output = model.output(image)
    println(output(0).data())
    println(output.length)
    println(output(0).rank())

    //val predictions = TrainedModels.VGG16.decodePredictions(output(0));
    //println(predictions)

    val imagNetLabels = new ImageNetLabels
    val predictions = imagNetLabels.decodePredictions(output(0))
    println(predictions)
    
    val modelSaveLocation = new File("Vgg-16.zip")
    ModelSerializer.writeModel(model, modelSaveLocation, true)
    
    println("--- Application end.---")
  }
}
