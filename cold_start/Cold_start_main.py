from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType
from pyspark.ml.regression import LinearRegression

from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
spark = SparkSession.builder.appName('tmp').getOrCreate()

# For training and hold out set
df=spark.read.parquet("final_val")
df=df.filter(df.rating!=0)
df,test=df.randomSplit([0.8,0.2],seed=100)
org=df


# get hold out set
print("Splitting the hold out set for cold start...")
book_id = df.select('book_id').distinct()
training_books,holdout_books=book_id.select("book_id").distinct().randomSplit([0.8,0.2],seed=100)
train = training_books.alias('df1').join(df.alias("df2"), training_books.book_id == df.book_id).select("df2.*")
holdout= holdout_books.alias('df1').join(df.alias("df2"), holdout_books.book_id == df.book_id).select("df2.*")

# read extra data set - genres from books
print("Reading extra information about book genres...")
k=spark.read.json("goodreads_book_genres_initial.json")
books_id_map=spark.read.parquet("books_id_mapping.parquet")

df=k.rdd.map(lambda x:[int(x.book_id)]+[int(i) if i else 0 for i in x.genres])\
    .toDF(["book_id","children", "comics_graphic", "fantasy_paranormal", 
           "fiction", "history_historical_fiction_biography", "mystery_thriller_crime", 
            "non_fiction", "poetry", "romance", "young_adult"])
df=df.join(books_id_map,books_id_map.real_book_id==df.book_id).drop("real_book_id").drop("book_id").withColumnRenamed("book_id_csv","book_id")
print(df.take(5))

# KNN model
#========================================================================================================
features=["children", "comics_graphic", "fantasy_paranormal", 
           "fiction", "history_historical_fiction_biography", "mystery_thriller_crime", 
            "non_fiction", "poetry", "romance", "young_adult"]
vecAssembler = VectorAssembler(inputCols=features, outputCol="features")
df_transform=vecAssembler.transform(df)
train_knn=df_transform.join(train,"book_id").select("book_id","features")
holdout_knn=df_transform.join(holdout,"book_id").select("book_id","features")

print("number of training books:",train_knn.count())# 587821
print("number of holdout books:",holdout_knn.count()) #136804  

# calculating distance
print("Calculating distance...")
brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0,
                                  numHashTables=3)
brp_model = brp.fit(train_knn)
print("Approximately geting neighbor on Euclidean distance smaller than 1.5:")
j=brp_model.approxSimilarityJoin(holdout_knn, train_knn, 1.5, distCol="EuclideanDistance")\
    .select(col("datasetA.book_id").alias("holdout_idA"),\
    col("datasetB.book_id").alias("train_idB"),\
    col("EuclideanDistance"))
print(j.take(5))

# training model
print("Training the recommender system on training data...")
train=train.withColumn("user_id", train["user_id"].cast(IntegerType()))\
    .withColumn("book_id", train["book_id"].cast(IntegerType()))\
    .withColumn("rating", train["rating"].cast(IntegerType()))
als = ALS(userCol='user_id',
            itemCol='book_id',
            ratingCol='rating',
              nonnegative=True
              )
model = als.fit(train)

#  get item and user matrix
print("Getting user and item matrix...")
ItemMatrix=model.itemFactors
UserMatrix=model.userFactors
productFeatures=model.itemFactors.rdd.map(lambda x: [x.id]+[i for i in x.features])\
    .toDF(["book_id"]+["features"+str(i) for i in range(10)])

# get cold start items latent factor representations
print("Getting cold start items latent factor representations using KNN...")
train_df=j.join(productFeatures,j.train_idB==productFeatures.book_id).drop("book_id")
gdf=train_df.groupBy("holdout_idA")
df_all=gdf.agg({"features0": "avg","features1":"avg","features2":"avg","features3":"avg",
                "features4": "avg","features5":"avg","features6":"avg","features7":"avg",
               "features8": "avg","features9":"avg"})\
        .withColumnRenamed("avg(features0)","features0").withColumnRenamed("avg(features1)","features1")\
        .withColumnRenamed("avg(features2)","features2").withColumnRenamed("avg(features3)","features3")\
        .withColumnRenamed("avg(features4)","features4").withColumnRenamed("avg(features5)","features5")\
        .withColumnRenamed("avg(features6)","features6").withColumnRenamed("avg(features7)","features7")\
        .withColumnRenamed("avg(features8)","features8").withColumnRenamed("avg(features9)","features9")

extra_items=df_all.rdd.map(lambda x:[x.holdout_idA]+[[x.features0,x.features1,x.features2,x.features3,x.features4,x.features5,
                                                     x.features6,x.features7,x.features8,x.features9]]).toDF(["book_id","features"])

# get new item matrix
print("Getting new item matrix...")
new_ItemMatrix=ItemMatrix.union(extra_items)
new_ItemMatrix=new_ItemMatrix.withColumn("id", new_ItemMatrix["id"].cast(IntegerType()))

# product of new item matrix and user matrix
print("Getting new data for recommender system...")
new_ItemMatrix=new_ItemMatrix.withColumnRenamed("id","book_id").withColumnRenamed("features","book_features")
UserMatrix=UserMatrix.withColumnRenamed("id","user_id").withColumnRenamed("features","user_features")

l=new_ItemMatrix.crossJoin(UserMatrix)
l=l.rdd.map(lambda x: [x.book_id,x.user_id]+[sum([i*j for i,j in zip(x.book_features,x.user_features)])])
l=l.toDF(["book_id","user_id","new_rating"])
print(l.take(5))

# TRAIN the new model
# ===========================================================================================
print("Training new model!...")
k1=holdout.select("book_id").distinct()
l=l.join(k1,"book_id")
k2=holdout.select("user_id").distinct()
l=l.join(k2,"user_id")
#print(new_ItemMatrix_holdout.count())# 25449  
#print(UserMatrix_holdout.count())# 5617
l=l.withColumnRenamed("user_id","user_id1").withColumnRenamed("book_id","book_id1")
l=l.take(100000)
final=holdout.join(l,(l.book_id1==holdout.book_id) &( l.user_id1==holdout.user_id)).select("book_id","user_id","new_rating")
final=final.withColumnRenamed("new_rating","rating")
#print(final.count())

train=train.select("user_id","book_id","rating")
final=final.union(train)
als = ALS(userCol='user_id',
            itemCol='book_id',
            ratingCol='rating',
              nonnegative=True
              )
final_model = als.fit(final)
final_model.write().overwrite().save("final_model_all")

# Prediction!
predictions = final_model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# compare with the full mode'
org=org.withColumn("user_id", df["user_id"].cast(IntegerType()))\
    .withColumn("book_id", df["book_id"].cast(IntegerType()))\
    .withColumn("rating", df["rating"].cast(IntegerType()))
als = ALS(userCol='user_id',
            itemCol='book_id',
            ratingCol='rating',
          coldStartStrategy='drop',
              nonnegative=True
              )
model = als.fit(org)


predictions =model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
