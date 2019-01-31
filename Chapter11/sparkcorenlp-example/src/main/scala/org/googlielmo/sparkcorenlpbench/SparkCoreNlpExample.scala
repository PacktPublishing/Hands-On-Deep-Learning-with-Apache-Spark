package org.googlielmo.sparkcorenlpbench

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import com.databricks.spark.corenlp.functions._
    
object SparkCoreNlpExample {
  def main(args: Array[String]): Unit = {

    val sparkSession = SparkSession
      .builder()
      .appName("spark-corenlp example")
      .master("local[*]")
      .getOrCreate()
      
    import sparkSession.implicits._
    
    val input = Seq(
      (1, "<xml>Packt is a publishing company based in Birmingham and Mumbai. It is a great publisher.</xml>")
    ).toDF("id", "text")
    
    val output = input
      .select(cleanxml('text).as('doc))
      .select(explode(ssplit('doc)).as('sen))
      .select('sen, tokenize('sen).as('words), ner('sen).as('nerTags), sentiment('sen).as('sentiment))
    
    output.show(truncate = false)
    
    sparkSession.stop()
  }
}