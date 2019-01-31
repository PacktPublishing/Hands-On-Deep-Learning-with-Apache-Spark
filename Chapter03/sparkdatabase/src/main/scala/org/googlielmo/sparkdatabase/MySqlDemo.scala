package org.googlielmo.sparkdatabase

import java.sql.DriverManager
import java.util.Properties
import org.apache.spark.sql.SparkSession

object MySqlDemo {
  @throws[Exception]
    def main(args: Array[String]): Unit = {
      var jdbcUsername = "myslus"
      var jdbcPassword = "your_password"
      
      Class.forName("com.mysql.jdbc.Driver")
      
      val jdbcHostname = "your_db_hostname_or_ip"
      val jdbcPort = 3306
      val jdbcDatabase ="sparkdb"
      // Create the JDBC URL without passing in the user and password parameters.
      val jdbcUrl = s"jdbc:mysql://${jdbcHostname}:${jdbcPort}/${jdbcDatabase}"

      // Create a Properties() object to hold the parameters.
      val connectionProperties = new Properties()
      connectionProperties.put("user", s"${jdbcUsername}")
      connectionProperties.put("password", s"${jdbcPassword}")
      
      val connection = DriverManager.getConnection(jdbcUrl, jdbcUsername, jdbcPassword)
      connection.isClosed()

      val spark = SparkSession
        .builder()
        .master("local[*]")
        .appName("Spark MySQL basic example")
        .getOrCreate()
        
      import spark.implicits._
      
      val jdbcDF = spark.read
        .format("jdbc")
        .option("url", jdbcUrl)
        .option("dbtable", s"${jdbcDatabase}.sparkexample")
        .option("user", jdbcUsername)
        .option("password", jdbcPassword)
        .load()
       
      jdbcDF.printSchema()
      println("Record count = " + jdbcDF.count())
      
      val filteredJDBC = jdbcDF.select("MerchantCountryCode", "TransactionAmountUSD")
                          .groupBy("MerchantCountryCode")
                          .avg("TransactionAmountUSD")
      filteredJDBC.collect.foreach { println }
      
      spark.close()
  }
}
