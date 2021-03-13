from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as func
import os
import sys
import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
import re
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator




# Create a SparkSession
#spark = SparkSession.builder.appName("SparkSQL").getOrCreate



spark = SparkSession.builder.appName("SparkSQL").getOrCreate()
read_file1=spark.read.option("header","true").option("inferSchema", "true").option("sep","\t").csv("title.basics.tsv") 
read_file2=spark.read.option("header","true").option("inferSchema", "true").option("sep","\t").csv("title.ratings.tsv") 



#there are some useless columns which we dont use in our analysis , so we drop them .
drop_cols1=['isAdult','startYear','endYear','runtimeMinutes']


df1 = read_file1.drop(*drop_cols1)
#read_file1=read_file1.select([column for column in read_file1.columns if column not in drop_cols1]).columns
#df1.show()

#df1.tconst=pd.Categorical(df1.tconst)
#df1["tconst_new"]=df1.tconst.cat.codes

df2 = df1.withColumn('tconst', regexp_replace('tconst', '^tt', '0'))
df3 = read_file2.withColumn('tconst', regexp_replace('tconst', '^tt', '0'))


#df3.show(10)


#print(type(df1["userID"]))


#df2 = df2.withColumn("tconst", df2["tconst"].cast('int'))
#df2.show(10)
print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
#print(df2.schema['tconst'].dataType)
df2 = df2.withColumn("tconst", df2["tconst"].cast('int'))
df3 = df3.withColumn("tconst", df3["tconst"].cast('int'))

#df3 = df3.withColumn("tconst",round(rand()*(1000-5)+5,0))







joined_table=df3.join(df2, ['tconst']).dropDuplicates()
#joined_table.show(10)

joined_table = joined_table.withColumn("tconst",round(rand()*(1000-5)+5,0))






#joined_table.datatypes()
#joined_table.filter(joined_table.tconst.isNotNull()).show()
#joined_table.show()

#Now we perform alternating least squares method , numVotes is not the only factor affecting the rating ,genre also plays a role , genre is the latent factor in our project.

als= ALS(maxIter=10, regParam=0.5, userCol="userID",itemCol="tconst", ratingCol="averageRating", nonnegative = True, implicitPrefs=False, coldStartStrategy="drop")

train, test = joined_table.randomSplit([0.8, 0.2])

#training the model
alsModel = als.fit(train)

#generating predictions
prediction = alsModel.transform(test)

#prediction.show(10) 


#Now we evaluate the model , and calculate mean square error


evaluator = RegressionEvaluator(metricName="mse", labelCol="averageRating",  predictionCol="prediction")
mse = evaluator.evaluate(prediction)
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#print(mse)

#our mse value is 2.25 which is a decent value , taking into consideration our dataset which was sparse and had NAN values.


#Now that our accuracy is not that bad , we will recommend top 3 movies to every user .

recommended_movie_df = alsModel.recommendForAllUsers(3)
recommended_movie_df.show(10, False)





















