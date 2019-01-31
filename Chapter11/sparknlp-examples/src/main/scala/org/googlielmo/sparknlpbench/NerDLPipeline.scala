package org.googlielmo.sparknlpbench

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

object NerDLPipeline {
  def main(args: Array[String]): Unit = {
     val sparkSession: SparkSession = SparkSession
        .builder()
        .appName("Ner DL Pipeline")
        .master("local[*]")
        .getOrCreate()
        
     import sparkSession.implicits._
     sparkSession.sparkContext.setLogLevel("WARN")   
        
     val document = new DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")
    
      val token = new Tokenizer()
        .setInputCols("document")
        .setOutputCol("token")
    
      val normalizer = new Normalizer()
        .setInputCols("token")
        .setOutputCol("normal")
    
      val ner = NerDLModel.pretrained()
        .setInputCols("normal", "document")
        .setOutputCol("ner")
    
      val nerConverter = new NerConverter()
        .setInputCols("document", "normal", "ner")
        .setOutputCol("ner_converter")
    
      val finisher = new Finisher()
        .setInputCols("ner", "ner_converter")
        .setIncludeMetadata(true)
        .setOutputAsArray(false)
        .setCleanAnnotations(false)
        .setAnnotationSplitSymbol("@")
        .setValueSplitSymbol("#")
    
      val pipeline = new Pipeline().setStages(Array(document, token, normalizer, ner, nerConverter, finisher))
    
      val testing = Seq(
        (1, "Packt is a famous publishing company"),
        (2, "Guglielmo is an author")
      ).toDS.toDF( "_id", "text")
    
      val result = pipeline.fit(Seq.empty[String].toDS.toDF("text")).transform(testing)
      Benchmark.time("Time to convert and show") {result.select("ner", "ner_converter").show(truncate=false)}
     
     sparkSession.stop()
  }
}