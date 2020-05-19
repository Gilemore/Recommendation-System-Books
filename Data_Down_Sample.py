'''
to run it in spark session:
spark-submit Data_Down_Sample.py yw4509 0.01
'''


from pyspark.sql import SparkSession

# pyspark --executor-memory 8g
# conf = (SparkConf()
#             .set('spark.executor.memory', '8g')
#             )
def down_sample(spark,input_path,output_path,percent):
    #read in the database
    df = spark.read.parquet(input_path)

    # discard users with few interactions, here we set the threshold to 10
    num_interactions = df.groupby("user_id").agg({"user_id": "count"}).withColumnRenamed("count(user_id)", "cnt")
    num_interactions_filtered = num_interactions.filter(num_interactions.cnt >= 10)

    # select unique user_id
    user_id = num_interactions_filtered.select('user_id').distinct()
    # down-sample 1% 10% or other percent based on input
    user_id_10 = user_id.rdd.sample(False, percent, seed=100).toDF()
    output = user_id_10.alias('df1').join(df.alias("df2"), user_id_10.user_id == df.user_id).select("df2.*")
    output.write.parquet(output_path)
    


if __name__=='__main__':
    
    spark = SparkSession.builder.appName('down_sample').getOrCreate()
    import sys
    #two input values for the netid and percent of downsampling
    id = sys.argv[1]
    percent = float(sys.argv[2]) #percent is the percentage we dowm sample, 0.1, 0.25...

    p = percent*100
    input_path = 'hdfs:/user/' + id + '/goodreads_interactions.parquet'
    output_path = 'hdfs:/user/' + id + '/small_reads_'+str(p)
    down_sample(spark,input_path,output_path,percent)
    # Transform the CSV file to parquet

    # # This loads the CSV file with proper header decoding and schema
    # df = spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv', header=True, 
    #                         schema='user_id INT,book_id INT,is_read INT,rating INT,is_reviewed INT')
    # pq_file_path = 'hdfs:/user/gw1107/goodreads.parquet'
    # # save the file as parguet
    # df.write.parquet(pq_file_path)
