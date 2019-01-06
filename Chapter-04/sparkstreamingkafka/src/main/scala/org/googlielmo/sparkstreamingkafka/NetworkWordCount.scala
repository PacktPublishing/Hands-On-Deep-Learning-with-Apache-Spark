package org.googlielmo.sparkstreamingkafka

import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}

object NetworkWordCount {
  def main(args: Array[String]) {
    if (args.length < 3) {
      System.err.println("Usage: NetworkWordCount <spark_master> <hostname> <port>")
      System.exit(1)
    }
    
    // Create the context with a 10 seconds batch size
    val sparkConf = new SparkConf().setAppName("NetworkWordCount").setMaster(args(0))
    val ssc = new StreamingContext(sparkConf, Seconds(10))
    
    // Create a socket stream on target ip:port and count the words in input stream of \n delimited text 
    val lines = ssc.socketTextStream(args(1), args(2).toInt, StorageLevel.MEMORY_AND_DISK_SER)
    val words = lines.flatMap(_.split(" "))
    val wordCounts = words.map(x => (x, 1)).reduceByKey(_ + _)
    wordCounts.print()
    ssc.start()
    ssc.awaitTermination()
  }
}