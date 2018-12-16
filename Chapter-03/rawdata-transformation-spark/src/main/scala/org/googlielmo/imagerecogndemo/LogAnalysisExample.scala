package org.googlielmo.imagerecogndemo

import java.io.{File, FileInputStream, FileOutputStream, IOException}
import java.net.URL
import java.util.zip.GZIPInputStream
import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.datavec.api.records.reader.impl.regex.RegexLineRecordReader
import org.datavec.api.transform.{ReduceOp, TransformProcess}
import org.datavec.api.transform.condition.ConditionOp
import org.datavec.api.transform.condition.column.LongColumnCondition
import org.datavec.api.transform.condition.string.StringRegexColumnCondition
import org.datavec.api.transform.filter.ConditionFilter
import org.datavec.api.transform.reduce.Reducer
import org.datavec.api.transform.schema.Schema
import org.datavec.api.util.ClassPathResource
import org.datavec.api.writable.IntWritable
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.datavec.spark.transform.{AnalyzeSpark, SparkTransformExecutor}
import org.joda.time.DateTimeZone

import scala.collection.JavaConverters._

object LogAnalysisExample {

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    // Spark setup
    val conf = new SparkConf
    conf.setMaster("local[*]")
    conf.setAppName("DataVec Log Analysis Example")
    val sc = new JavaSparkContext(conf)

    // Specify a schema for the data. 
    val schema = new Schema.Builder()
      .addColumnString("host")
      .addColumnString("timestamp")
      .addColumnString("request")
      .addColumnInteger("httpReplyCode")
      .addColumnInteger("replyBytes")
      .build

    // Load the data.
    val directory = new ClassPathResource("access_log").getFile.getAbsolutePath 
    println(directory)
    var logLines = sc.textFile(directory)
    // Remove the invalid rows.
    logLines = logLines.filter { (s: String) =>
      s.matches("(\\S+) - - \\[(\\S+ -\\d{4})\\] \"(.+)\" (\\d+) (\\d+|-)") 
    }
    
    // Parse the input data using a RegexLineRecordReader. 
    val regex = "(\\S+) - - \\[(\\S+ -\\d{4})\\] \"(.+)\" (\\d+) (\\d+|-)"
    val rr = new RegexLineRecordReader(regex, 0)
    val parsed = logLines.map(new StringToWritablesFunction(rr))
    
    // Check the data quality.
    val dqa = AnalyzeSpark.analyzeQuality(schema, parsed)
    println("----- Data Quality -----")
    println(dqa) 

    // Specify the transformation to do.
    val tp: TransformProcess = new TransformProcess.Builder(schema)
      .conditionalReplaceValueTransform("replyBytes", new IntWritable(0), new StringRegexColumnCondition("replyBytes", "\\D+"))
      .stringToTimeTransform("timestamp", "dd/MMM/YYYY:HH:mm:ss Z", DateTimeZone.forOffsetHours(-8))
      .reduce(new Reducer.Builder(ReduceOp.CountUnique)
        .keyColumns("host")                             
        .countColumns("timestamp")                      
        .countUniqueColumns("request", "httpReplyCode") 
        .sumColumns("replyBytes")                       
        .build
      )

      // Filter out all hosts that requested less than 1 million bytes in total.
      .filter(new ConditionFilter(new LongColumnCondition("sum(replyBytes)", ConditionOp.LessThan, 1000000)))
      .build

    val processed = SparkTransformExecutor.execute(parsed, tp)
    processed.cache
    
    
    // Perform analysis on the final Data.
    val finalDataSchema = tp.getFinalSchema
    val finalDataCount = processed.count
    val sample = processed.take(10)

    val analysis = AnalyzeSpark.analyze(finalDataSchema, processed)

    sc.stop()
    Thread.sleep(4000) 

    // Display the result of the analysis
    println("----- Final Data Schema -----")
    println(finalDataSchema)

    println("\n\nFinal data count: " + finalDataCount)

    println("\n\n----- Samples of final data -----")
    for (l <- sample.asScala) {
      println(l)
    }

    println("\n\n----- Analysis -----")
    println(analysis)
  }
  
}
