package org.googlielmo.sparkstreamingkafka

import kafka.serializer.StringDecoder

import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka._
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.schema.Schema
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.{StringToWritablesFunction, WritablesToStringFunction}
  
import scala.collection.JavaConverters._

object DirectKafkaDataVec {
  def main(args: Array[String]) {
    // Check for the correct number of arguments
    if (args.length < 3) {
      System.err.println(s"""
        |Usage: DirectKafkaWordCount <spark_master> <brokers> <topics>
        |  <spark_master> is the Spark master URL
        |  <brokers> is a list of one or more Kafka brokers
        |  <topics> is a list of one or more kafka topics to consume from
        |
        """.stripMargin)
      System.exit(1)
    }
    
    val Array(master, brokers, topics) = args
    
    // Define the input data schema
    val inputDataSchema = new Schema.Builder()
        .addColumnsString("id", "description", "notes")
        .build
      
    println(inputDataSchema)
    
    // Define some transformation (remove some columns)
    val tp = new TransformProcess.Builder(inputDataSchema)
        .removeColumns("notes")
        .build
        
    // Get and then print the new schema (after the transformations)  
    val outputSchema = tp.getFinalSchema
    println("\n\n\nSchema after transforming data:")
    println(outputSchema)
      
    // Create a streaming context with 5 seconds batch interval
    val sparkConf = new SparkConf().setAppName("DirectKafkaDataVec").setMaster(master)
    val ssc = new StreamingContext(sparkConf, Seconds(5))
    ssc.sparkContext.setLogLevel("WARN")
    
    // Create a direct Kafka stream
    val topicsSet = topics.split(",").toSet
    val kafkaParams = Map[String, String]("metadata.broker.list" -> brokers)
    val messages = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](
      ssc, kafkaParams, topicsSet)
      
    // Get the lines
    val lines = messages.map(_._2)
    
    lines.foreachRDD { rdd => 

      val javaRdd = rdd.toJavaRDD
      val rr = new CSVRecordReader
      val parsedInputData = javaRdd.map(new StringToWritablesFunction(rr))
      
      if(!parsedInputData.isEmpty) {
        val processedData = SparkTransformExecutor.execute(parsedInputData, tp)
        
        // Collect the data locally and print it
        val processedAsString = processedData.map(new WritablesToStringFunction(","))
        val processedCollected = processedAsString.collect
        val inputDataCollected = javaRdd.collect
        
        println("\n\n---- Original Data ----")
        for (s <- inputDataCollected.asScala) println(s)
  
        println("\n\n---- Processed Data ----")
        for (s <- processedCollected.asScala) println(s)
      }
    }
    
    // Start the computation and keep it alive waiting for a termination signal
    ssc.start()
    ssc.awaitTermination()
  }
}