from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FirstExample").getOrCreate()

df = spark.read.csv("./data/GiveMeSomeCredit/cs-training.csv")
print(df.show(5))
