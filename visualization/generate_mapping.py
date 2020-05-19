from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
									_kl_divergence)
# from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# # We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
				rc={"lines.linewidth": 2.5})

# # We'll generate an animation with matplotlib and moviepy.
# from moviepy.video.io.bindings import mplfig_to_npimage
# import moviepy.editor as mpy
import argparse

def scatter(x, colors):
	# We choose a color palette with seaborn.
	palette = np.array(sns.color_palette("hls", 10))

	# We create a scatter plot.
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')
	sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,c=palette[colors.astype(np.int)])
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')

	# We add the labels for each digit.
	txts = []
	for i in range(10):
		# Position of each label.
		xtext, ytext = np.median(x[colors == i, :], axis=0)
		txt = ax.text(xtext, ytext, mapping[i], fontsize=24)
		txt.set_path_effects([
			PathEffects.Stroke(linewidth=5, foreground="w"),
			PathEffects.Normal()])
		txts.append(txt)

	return f, ax, sc, txts


if __name__=='__main__':
	# read from the save models to generate mapping
	'''
	--modelpath 'hdfs:/user/yw4509/best_model100.0_version2.0'
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--modelpath', default='hdfs:/user/yw4509/best_model100.0_version2.0',
						help='saved model')
	args = parser.parse_args()
	modelpath = args.modelpath
	print(modelpath)
	spark = SparkSession.builder.appName('map').getOrCreate()

	# load model
	model = ALSModel.load(modelpath)
	modelrank = model.rank
	# get book representation from the model
	model_book = model.itemFactors

	filepath = 'hdfs:/user/gw1107/book_id_map.csv'
	book_id_convert  = spark.read.csv(filepath,header=True)
	# book_id_convert = spark.createDataFrame(item)
	filepath = 'hdfs:/user/gw1107/book_with_genre.csv'
	book_w_genre = spark.read.csv(filepath,header=True)
	# book_w_genre = spark.createDataFrame(book_w_genre)

	book_w_genre.createOrReplaceTempView('book_w_genre')
	book_id_convert.createOrReplaceTempView('book_id_convert')
	model_book.createOrReplaceTempView('model_item')
	query = '''
		SELECT a.id as bid_model, 
		c.book_id as bid_real, 
		a.features as features,
		c.genres as genres 
		FROM model_item a
		JOIN book_id_convert b ON a.id = b.book_id_csv
		JOIN book_w_genre c ON b.book_id = c.book_id
		'''
	df_sql = spark.sql(query)
	df = df_sql.select("*").toPandas()

	l=[]
	for r in df['features']:
		l.append([*r])
	model_item_feature =pd.DataFrame(l)
	final_df =  pd.concat([df, model_item_feature],axis=1)[['bid_real','bid_model','genres']+list(range(modelrank))]
	final_df = final_df.dropna(axis=0)
	# # label encode
	# le = LabelEncoder()
	# final_df['label'] = le.fit_transform(list(final_df.genres))
	# mapping = dict(zip( range(len(le.classes_)),le.classes_))
	# len_label = len(le.classes_)
	# save final_df
	filepath = './'+modelpath.split('/')[-1]
	final_df.to_csv(filepath+'final_df.csv')

	# # We first reorder the data points according to the labels
	# X = np.vstack([final_df[list(range(modelrank))][final_df.label==i]
	# 			   for i in range(len_label)])
	# y = np.hstack([final_df.label[final_df.label==i]
	# 		   for i in range(len_label)])
	# digits_proj = TSNE(random_state=RS).fit_transform(X)
	
	# f, ax, sc, txts = scatter(digits_proj, y)
	# f.savefig(filepath+'figure.png')
	
