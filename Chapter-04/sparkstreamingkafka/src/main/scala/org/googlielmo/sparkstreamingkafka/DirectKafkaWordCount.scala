package org.googlielmo.sparkstreamingkafka

import kafka.serializer.StringDecoder

import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka._

object DirectKafkaWordCount {
  def main(args: Array[String]) {
    // Check for the correct number of arguments
    if (args.length < 3) {
      System.err.println(s"""
        |Usage: DirectKafkaWordCount <spark_master> <brokers> <topics>
        |  <spark_master> is the Spark Master URL
        |  <brokers> is a list of one or more Kafka brokers
        |  <topics> is a list of one or more kafka topics to consume from
        |
        """.stripMargin)
      System.exit(1)
    }
    
    val Array(master, brokers, topics) = args
    
    // Create a streaming context with 5 seconds batch interval
    val sparkConf = new SparkConf().setAppName("DirectKafkaWordCount").setMaster(master)
    val ssc = new StreamingContext(sparkConf, Seconds(5))
    ssc.sparkContext.setLogLevel("WARN")
    
    // Create a direct Kafka stream
    val topicsSet = topics.split(",").toSet
    val kafkaParams = Map[String, String]("metadata.broker.list" -> brokers)
    val messages = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](
      ssc, kafkaParams, topicsSet)
      
    // Get the lines, split them into words and then count the words and print to console
    val lines = messages.map(_._2)
    val words = lines.flatMap(_.split(" "))
    val wordCounts = words.map(x => (x, 1L)).reduceByKey(_ + _)
    wordCounts.print()
    
    // Start the computation and keep it alive waiting for a termination signal
    ssc.start()
    ssc.awaitTermination()
  }
}