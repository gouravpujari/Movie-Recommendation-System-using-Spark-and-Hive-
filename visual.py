
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as func
import os
import sys
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
plt.style.use("seaborn-pastel")




spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

read_file1=spark.read.option("header","true").option("inferSchema", "true").option("sep","\t").csv("file:/home/gou/imdb/title.basics.tsv") 
read_file2=spark.read.option("header","true").option("inferSchema", "true").option("sep","\t").csv("file:/home/gou/imdb/title.ratings.tsv") 

#df_title_basics.show(10)

drop_cols1=['isAdult','startYear','endYear','runtimeMinutes']
df1 = read_file1.drop(*drop_cols1)

df2 = df1.withColumn('tconst', regexp_replace('tconst', '^tt', '0'))
df3 = read_file2.withColumn('tconst', regexp_replace('tconst', '^tt', '0'))


df2 = df2.withColumn("tconst", df2["tconst"].cast('int'))
df3 = df3.withColumn("tconst", df3["tconst"].cast('int'))

joined_table=df3.join(df2, ['tconst']).dropDuplicates()

joined_table = joined_table.withColumn("tconst",round(rand()*(1000-5)+5,0))


#df_title_basics.titleType.value_counts().plot.pie(autopct="%.0f%%",figsize=(6,6),pctdistance=0.8,wedgeprops=dict(width=0.4))
#plt.show()

group = joined_table.groupby("tconst").avg("averageRating")
#group.show()
group = group.limit(5)




#df_title_basics.plot.pie(y='titleType', figsize=(5,5),labels=df_['Name'])
x=group.toPandas()['tconst'].values.tolist()
y=group.toPandas()['avg(averageRating)'].values.tolist()
plt.bar(x,y)
plt.show()
	
#spark.stop()



