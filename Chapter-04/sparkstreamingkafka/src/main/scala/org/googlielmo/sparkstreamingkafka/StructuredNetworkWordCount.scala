package org.googlielmo.sparkstreamingkafka

import org.apache.spark.sql.SparkSession

object StructuredNetworkWordCount {
  def main(args: Array[String]) {
    if (args.length < 3) {
      System.err.println("Usage: StructuredNetworkWordCount <spark_master> <hostname> <port>")
      System.exit(1)
    }
    
    val host = args(1)
    val port = args(2).toInt

    val spark = SparkSession
      .builder
      .appName("StructuredNetworkWordCount")
      .master(args(0))
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    
    import spark.implicits._

    // Create a DataFrame representing the stream of input lines from connection to host:port
    val lines = spark.readStream
      .format("socket")
      .option("host", host)
      .option("port", port)
      .load()

    // Split the DataFrame lines into words
    val words = lines.as[String].flatMap(_.split(" "))

    // Generate the word count
    val wordCounts = words.groupBy("value").count()

    // Start running the query that prints the running counts to the console
    val query = wordCounts.writeStream
      .outputMode("complete")
      .format("console")
      .start()

    query.awaitTermination()
  }
}