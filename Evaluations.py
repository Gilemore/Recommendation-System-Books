from pyspark.sql import functions as F
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import Row
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import explode, col, mean
import argparse
from collections import defaultdict
import pickle

'''
spark-submit Evaluations.py --account yw4509 --percent 10.0 --rank "10" --maxIter "20" --regParam "0.17"
'''


def evaluateTopk(model,data,top_k=500):
    '''
    Input:
    validation: RDD
        - user, product (book_id), rating
    '''
    truth=spark.createDataFrame(data).groupby("user").agg(F.collect_set("product"))
    print("Getting Predictions...")
    tmp1=model.recommendProductsForUsers(top_k).map(lambda r: [r[0],[k.product for k in r[1]]])
    predictions=spark.createDataFrame(tmp1,["user","predictions"])


    print("Predictions and Labels...")
    k=predictions.join(truth,truth.user==predictions.user)
    final=k.rdd.map(lambda r: [r[1],r[3]])
    metrics=RankingMetrics(final)

    print("\nCalculate NDCG at {}...".format(top_k))
    res1=metrics.ndcgAt(top_k)
    print("NDCG at {}: {}".format(top_k,res1))

    print("\nCalculate MAP...")
    res2=metrics.meanAveragePrecision
    print("MAP: {}".format(res2))

    print("\nCalculate Precision at {}...".format(top_k))
    res3=metrics.precisionAt(top_k)
    print("Precision at {}: {}".format(top_k,res1))

    return res1,res2,res3
if __name__=='__main__':
    # read from the acount hdfs, of the four parquet files we need for training and evaluation
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--account', default='yw4509',
                        help='account name to store data after split')
    parser.add_argument('--percent', default='10.0',
                        help='percent of downsampling')
    parser.add_argument('--local', default="F",
                        help='option for running at local or hdfs')
    parser.add_argument('--rank', default="10",
                        help='tuning parameter in ALS, rank')
    parser.add_argument('--maxIter', default="20",
                        help='tuning parameter in ALS, maxIter')
    parser.add_argument('--regParam', default="0.17",
                        help='tuning parameter in ALS, regularization parameter')
    #
    args = parser.parse_args()
    account = args.account
    percent=args.percent

    path_hdfs = 'hdfs:/user/{}/'.format(account)

    spark = SparkSession.builder.appName('evaluations').getOrCreate()

    training_val = spark.read.parquet(path_hdfs+'final_train_val'+percent)
    training_test = spark.read.parquet(path_hdfs+'final_train_test'+percent)
    val = spark.read.parquet(path_hdfs+'final_val'+percent)
    test = spark.read.parquet(path_hdfs+'final_test'+percent)

    train=training_val.rdd.map(lambda l: Rating(int(l.book_id), int(l.user_id), float(l.rating)))
    validation=val.rdd.map(lambda l: Rating(int(l.book_id), int(l.user_id), float(l.rating)))
    test=test.rdd.map(lambda l: Rating(int(l.book_id), int(l.user_id), float(l.rating)))

    rank = int(args.rank)
    numIterations = int(args.maxIter)
    lambda_=float(args.regParam)
    
    model = ALS.train(train,rank=rank,iterations=numIterations,lambda_=lambda_)

    print("Estimating validation...")
    res1,res2,res3=evaluateTopk(model,validation,top_k=500)
    print("Estimating test...")
    res4,res5,res6=evaluateTopk(model,test,top_k=500)
    print("Finishing...")
    res.append([[res1,res2,res3],[res4,res5,res6]])
    outputfile=path_hdfs+"res"+percent+".pkl"
    res = pickle.load(open(outputfile,"rb"))
    
