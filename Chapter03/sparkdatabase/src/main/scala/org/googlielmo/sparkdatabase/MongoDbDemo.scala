package org.googlielmo.sparkdatabase

import org.apache.spark.sql.SparkSession
import com.mongodb.spark._

case class Transaction(CustomerID: String, 
                      MerchantID: String,
                      MerchantCountryCode: String,
                      DateTimeString: String,
                      NumItemsInTransaction: Int,
                      TransactionAmountUSD: Double,
                      FraudLabel: String)

object MongoDbDemo {
  @throws[Exception]
    def main(args: Array[String]): Unit = {
      val sparkSession = SparkSession.builder()
      .master("local")
      .appName("MongoSparkConnectorIntro")
      .config("spark.mongodb.input.uri", "mongodb://host_or_ip/sparkmdb.sparkexample")
      .config("spark.mongodb.output.uri", "mongodb://host_or_ip/sparkmdb.sparkexample")
      .getOrCreate()
      
      val df = MongoSpark.load(sparkSession) 
      df.printSchema()

      df.collect.foreach { println }
      
      val transactions = MongoSpark.load[Transaction](sparkSession)
      transactions.createOrReplaceTempView("transactions")

      val filteredTransactions  = sparkSession.sql("SELECT CustomerID, MerchantID FROM transactions WHERE TransactionAmountUSD = 100")
      filteredTransactions .show
      
      sparkSession.close()
  }
}
