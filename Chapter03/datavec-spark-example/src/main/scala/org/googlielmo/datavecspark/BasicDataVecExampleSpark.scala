package org.googlielmo.datavecspark

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.condition.ConditionOp
import org.datavec.api.transform.condition.column.CategoricalColumnCondition
import org.datavec.api.transform.filter.ConditionFilter
import org.datavec.api.transform.schema.Schema
import org.datavec.api.util.ClassPathResource
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.{StringToWritablesFunction, WritablesToStringFunction}

import java.util.Arrays
import java.util.HashSet

import scala.collection.JavaConverters._

object BasicDataVecExampleSpark {
    @throws[Exception]
    def main(args: Array[String]): Unit = {
      //----- Transformation section ---
      // Define the input data schema
      val inputDataSchema = new Schema.Builder()
         //We can define a single column
        .addColumnString("DateTimeString")
         //Or for convenience define multiple columns of the same type
        .addColumnsString("CustomerID", "MerchantID")
        //We can define different column types for different types of data:
        .addColumnInteger("NumItemsInTransaction")
        .addColumnCategorical("MerchantCountryCode", List("USA", "CAN", "FR", "MX").asJava)
         //Some columns have restrictions on the allowable values, that we consider valid:
        .addColumnDouble("TransactionAmountUSD", 0.0, null, false, false) //$0.0 or more, no maximum limit, no NaN and no Infinite values
        .addColumnCategorical("FraudLabel", List("Fraud", "Legit").asJava)
        .build
        
      // Print out the schema before the transformation(s)
      println("Input data schema details:")
      println(inputDataSchema)
      
      // Define some transformation
      // Remove some columns
      val tp = new TransformProcess.Builder(inputDataSchema)
        .removeColumns("CustomerID", "MerchantID")
        .filter(new ConditionFilter(
                    new CategoricalColumnCondition("MerchantCountryCode", ConditionOp.NotInSet, new HashSet(Arrays.asList("USA","CAN")))))
        .build
       
      // Get and then print the new schema (after the transformations)  
      val outputSchema = tp.getFinalSchema
      println("\n\n\nSchema after transforming data:")
      println(outputSchema)
      
      val conf = new SparkConf
      conf.setMaster("local[*]")
      conf.setAppName("DataVec Example")
      
      val sc = new JavaSparkContext(conf)
      
      // Read the data
      val directory = new ClassPathResource("exampledata.csv").getFile.getAbsolutePath 
      println(directory)
      val stringData = sc.textFile(directory)
      
      // Parse the CSV file
      val rr = new CSVRecordReader
      val parsedInputData = stringData.map(new StringToWritablesFunction(rr))

      // Execute the transformations defined earlier
      val processedData = SparkTransformExecutor.execute(parsedInputData, tp)

      // Collect the data locally and print it
      val processedAsString = processedData.map(new WritablesToStringFunction(","))
      val processedCollected = processedAsString.collect
      val inputDataCollected = stringData.collect

      println("\n\n---- Original Data ----")
      for (s <- inputDataCollected.asScala) println(s)

      println("\n\n---- Processed Data ----")
      for (s <- processedCollected.asScala) println(s)

      sc.close()
    }
}
