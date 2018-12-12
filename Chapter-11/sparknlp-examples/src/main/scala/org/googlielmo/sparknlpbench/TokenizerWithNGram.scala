package org.googlielmo.sparknlpbench

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.SparkSession

object TokenizerWithNGram {
  def main(args: Array[String]): Unit = {
    val sparkSession: SparkSession = SparkSession
    .builder()
    .appName("Tokenize with n-gram example")
    .master("local[*]")
    .config("spark.driver.memory", "1G")
    .config("spark.kryoserializer.buffer.max","200M")
    .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
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
  
    val finisher = new Finisher()
      .setInputCols("normal")
  
    val ngram = new NGram()
      .setN(3)
      .setInputCol("finished_normal")
      .setOutputCol("3-gram")
  
    val gramAssembler = new DocumentAssembler()
      .setInputCol("3-gram")
      .setOutputCol("3-grams")
  
    val pipeline = new Pipeline().setStages(Array(document, token, normalizer, finisher, ngram, gramAssembler))
  
    val testing = Seq(
      (1, "Packt is a famous publishing company"),
      (2, "Guglielmo is an author")
    ).toDS.toDF( "_id", "text")
  
    val result = pipeline.fit(Seq.empty[String].toDS.toDF("text")).transform(testing)
    Benchmark.time("Time to convert and show") {result.show(truncate=false)}
    
    sparkSession.stop
  }
}