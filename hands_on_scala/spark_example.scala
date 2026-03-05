package com.sparkbyexamples.spark.rdd

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object WordCountExample {
  def main(arg: Array[String]) =
    val spark: SparkSession = SparkSession
      .builder()
      .master("local[3]")
      .appName("SparkByExamples.com")
      .getOrCreate()
    val sc = spark.sparkContext

    val rdd = sc.textFile("./test.txt")
    println("initial particion count: " + rdd.getNumPartitions)
}
