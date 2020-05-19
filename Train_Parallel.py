from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from multiprocessing.pool import ThreadPool
import numpy as np
import pandas as pd
import os

import argparse
from collections import defaultdict

from pyspark.sql import functions as F
from pyspark.sql import Row

'''
spark-submit Train_Parallel.py --account 'yw4509' --local 'F' --pram_rank "50,70,90,100,120" --pram_maxIter "10" --pram_regParam "0.01,0.03,0.05,0.10,0.20,0.30" --percent '10.0'
'''

def _parallelFitTasks(est, train, eva, validation, epm):
    """
    Creates a list of callables which can be called from different threads to fit and evaluate
    an estimator in parallel. Each callable returns an `(index, metric)` pair.
    :param est: Estimator, the estimator to be fit.
    :param train: DataFrame, training data set, used for fitting.
    :param eva: Evaluator, used to compute `metric`
    :param validation: DataFrame, validation data set, used for evaluation.
    :param epm: Sequence of ParamMap, params maps to be used during fitting & evaluation.
    :return: (int, float), an index into `epm` and the associated metric value.
    """
    modelIter = est.fitMultiple(train, epm)
    def singleTask():
        index, model = next(modelIter)
        model1 = model.transform(validation, epm[index])
        metric = eva.evaluate(model1)
        return index, metric
    return [singleTask] * len(epm)

def tuning_als(model,train, validation, ParamMaps,percent,version,top_k=500):
    """
     Validation for hyper-parameter tuning.
     param_dict is a dict for rank, max_iter, reg_param
     Given training data and validation data, uses evaluation metric on the validation set to select the best model.
     Evaluate on the precision of the Top 500 prediction
    """
    max_precision = -1
    # can be paralleled
    # p = Params()
    est=model
    epm = ParamMaps
    numModels = len(epm)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol='rating',
                              predictionCol="prediction")
    eva = evaluator
    metrics = [0.0] * numModels
    # matrix_metrics = [[0 for x in range(1)] for y in range(len(epm))]

    pool = ThreadPool(processes=numModels)
    tasks = _parallelFitTasks(est, train, eva, validation, epm)

    for j, metric in pool.imap_unordered(lambda f: f(), tasks):
        print('This is valuation for ', j, 'th round in the grid serach')
        metrics[j] = metric

    bestIndex = np.argmin(metrics)
    best_model = est.fit(train, epm[bestIndex])
    '''
    For write file purposes:
    '''
    l1 = 'Metrics:'+str(metrics)
    l2 = 'best Index' + str( bestIndex)
    l3 = 'best metrics' + str (min(metrics))
    l4 = 'The best parameter is:'+ str(epm[bestIndex])
    l5 = 'The whole param map is:' + str(epm)
    data = {'metrics': l1, 'index': l2, 'best metrics': l3, 'best param': l4, 'map': l5}
    df = pd.Series(data).to_frame()
    df_s = spark.createDataFrame(df)
    f_name = "out" + percent +'_version' + version +".csv"
    df_s.write.csv('hdfs:/user/yw4509/'+f_name)

    print('Metrics:', metrics,'best Index',bestIndex, 'best metrics', min(metrics))
    print('The best parameter is:', epm[bestIndex])
    print('The whole param map is:',epm)
    return best_model, bestIndex

if __name__=='__main__':
    # read from the acount hdfs, of the four parquet files we need for training and evaluation
    '''
    --account 'gw1107' --local 'T' --pram_rank "12,13,14" --pram_maxIter "18,19,20" --pram_regParam "0.17,0.18,0.19"
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--account', default='gw1107',
                        help='account name to store data after split')
    parser.add_argument('--local', default="F",
                        help='option for running at local or hdfs')
    parser.add_argument('--pram_rank', default="10,15,20",
                        help='tuning parameter in ALS, rank')
    parser.add_argument('--pram_maxIter', default="10,15,20",
                        help='tuning parameter in ALS, maxIter')
    parser.add_argument('--pram_regParam', default="0.1,0.15,0.20",
                        help='tuning parameter in ALS, regularization parameter')
    parser.add_argument('--percent', default="10.0",
                        help='percentage of data which is based on')
    parser.add_argument('--version', default="1.0",
                        help='version of the tuning')
    #
    args = parser.parse_args()
    # path of the hdfs that store the split data
    account = args.account
    local = args.local
    percent = args.percent
    version = args.version
    pram_rank = [int(e) for e in args.pram_rank.split(',')]
    pram_maxIter = [int(e) for e in args.pram_maxIter.split(',')]
    pram_regParam = [float(e) for e in args.pram_regParam.split(',')]

    # The following is for debug mainly at local,
    #comment the parse section above and uncomment the following part for local debugger:
#     account = 'yw4509'
#     local = 'T'
#     pram_rank = [int(e) for e in ["12","15"]]
#     pram_maxIter = [int(e) for e in ["10"]]
#     pram_regParam = [float(e) for e in ["0.1","0.15"]]


    path_hdfs = 'hdfs:/user/{}/'.format(account)
    path_local = '/Users/yunya/1004/Project/percent1/'

    spark = SparkSession.builder.appName('train').getOrCreate()
    # read data from the account hdfs
    file1 = 'final_train_val'+percent
    file2 = 'final_train_test'+percent
    file3 = 'final_val'+percent
    file4 = 'final_test'+percent
    if local == 'F':
        training_val = spark.read.parquet(path_hdfs+file1)
        training_test = spark.read.parquet(path_hdfs+file2)
        val = spark.read.parquet(path_hdfs+file3)
        test = spark.read.parquet(path_hdfs+file4)
    if local == 'T':
        training_val = spark.read.parquet(path_local+file1)
        training_test = spark.read.parquet(path_local+file2)
        val = spark.read.parquet(path_local+file3)
        test = spark.read.parquet(path_local+file4)

    als = ALS(userCol='user_id',
                itemCol='book_id',
                ratingCol='rating',
                coldStartStrategy='drop',
                nonnegative=True)
    ParamMaps = param_grid = ParamGridBuilder()\
                .addGrid(als.rank,pram_rank)\
                .addGrid(als.maxIter,pram_maxIter)\
                .addGrid(als.regParam,pram_regParam)\
                .build()
    print("Start Tuning...")
    best_model, bestIndex = tuning_als(als,training_val, val, ParamMaps, percent,version,top_k=500)
    if local =='T':
        best_model_path = path_local +'best_model'+percent + '_version' + version
    else:
        best_model_path = path_hdfs +'best_model'+percent + '_version' + version
    best_model.write().overwrite().save(best_model_path)
    print("Tuning finished...")



