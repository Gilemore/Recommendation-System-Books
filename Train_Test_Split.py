'''
to run it in spark session:
spark-submit Train_Test_Split.py yw4509 small_reads_25.0 25.0
'''


from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

def train_test_split(spark, input_path, percent,id):
    df = spark.read.parquet(input_path)

    print("change the data type and remove NA values in user_id, book_id and rating")

    df = df.withColumn("user_id", df["user_id"].cast(IntegerType())) \
        .withColumn("book_id", df["book_id"].cast(IntegerType())) \
        .withColumn("is_read", df["is_read"].cast(IntegerType())) \
        .withColumn("rating", df["rating"].cast(IntegerType())) \
        .withColumn("is_reviewed", df["is_reviewed"].cast(IntegerType()))

    df = df.filter(df.user_id.isNotNull())\
        .filter(df.book_id.isNotNull())\
        .filter(df.rating.isNotNull())

    print(df.printSchema())

    
    print("Starting Splitting...")

    user_id = df.select('user_id').distinct()
    training_users,val_users, test_users=user_id.select("user_id").distinct().randomSplit([0.6,0.2,0.2],seed=100)

    print("For training")

    train = training_users.alias('df1').join(df.alias("df2"), training_users.user_id == df.user_id).select("df2.*")
    val = val_users.alias('df1').join(df.alias("df2"), val_users.user_id == df.user_id).select("df2.*")
    test = test_users.alias('df1').join(df.alias("df2"), test_users.user_id == df.user_id).select("df2.*")

    print("For validation...")

    val.createOrReplaceTempView('val')
    val_index = spark.sql('''select user_id, book_id,is_read, rating, is_reviewed,
                        row_number() over(partition by user_id order by book_id) as idx
                        from val order by user_id''')
    val_index.createOrReplaceTempView('val_index')
    val_index_v2 = spark.sql('''select user_id, book_id, is_read, rating, is_reviewed,
                       idx, avg(idx) over(partition by user_id) as avg
                        from val_index''')
    val_index_v3 = val_index_v2.withColumn('flag', F.when(F.col("idx") <= F.col('avg'), 1).otherwise(2))
    val_val = val_index_v3.filter(val_index_v3.flag == 2)
    val_train = val_index_v3.filter(val_index_v3.flag == 1)

    print("For testing...")


    test.createOrReplaceTempView('test')
    test_index = spark.sql('''select user_id, book_id,is_read, rating, is_reviewed,
                        row_number() over(partition by user_id order by book_id) as idx
                        from test order by user_id''')
    test_index.createOrReplaceTempView('test_index')
    test_index_v2 = spark.sql('''select user_id, book_id, is_read, rating, is_reviewed,
                       idx, avg(idx) over(partition by user_id) as avg
                        from test_index''')
    test_index_v3 = val_index_v2.withColumn('flag', F.when(F.col("idx") <= F.col('avg'), 1).otherwise(2))
    test_val = test_index_v3.filter(test_index_v3.flag == 2)
    test_train = test_index_v3.filter(test_index_v3.flag == 1)

    columns_to_drop = ['idx', 'avg','flag']
    val_val = val_val.drop(*columns_to_drop)
    val_train = val_train.drop(*columns_to_drop)
    test_val = test_val.drop(*columns_to_drop)
    test_train = test_train.drop(*columns_to_drop)

    print("Final Union...")

    final_train_val = train.union(val_train)
    final_train_test = train.union(test_train)

    path_1 = 'hdfs:/user/'+id+'/final_train_val'+ percent
    path_2 = 'hdfs:/user/'+id+'/final_train_test'+percent
    path_3 = 'hdfs:/user/'+id+'/final_val'+percent
    path_4 = 'hdfs:/user/'+id+'/final_test'+percent

    final_train_val.write.mode('overwrite').parquet(path_1)
    final_train_test.write.mode('overwrite').parquet(path_2)
    val_val.write.mode('overwrite').parquet(path_3)
    test_val.write.mode('overwrite').parquet(path_4)

#['user_id', 'book_id', 'is_read', 'rating', 'is_reviewed']

if __name__=='__main__':
    spark = SparkSession.builder.appName('split').getOrCreate()

    import sys
    # two input values for the netid and input file
    id = sys.argv[1]
    file_name = sys.argv[2] #small_reads_1.0
    percent = sys.argv[3] #100.0 25.0 10.0 1.0

    input_path='hdfs:/user/'+ id +'/'+ file_name

    train_test_split(spark, input_path, percent,id)




