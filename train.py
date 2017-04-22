"""
LSTM modeling of discourse relation senses
author: Samuel Ronnqvist <sronnqvi@abo.fi>
source: https://github.com/sronnqvist/discourse-ablstm
"""
import os
# Set hash seed for python3 to produce deterministicly orderered list of keys from dict
os.environ['PYTHONHASHSEED'] = "0"
# Switch off non-deterministic operations (optimization) in theano (did not produce fully deterministic results!)
#os.environ['THEANO_FLAGS'] = "dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic"
import numpy as np
import random
# Set random seeds to reduce variability of results
random.seed(123)
np.random.seed(123)
import json
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import WeightRegularizer # for keras 1
#from keras.regularizers import l2 # for keras 2
from keras.constraints import maxnorm
from keras import optimizers
import FarrokhAttentionLayer as F
import resources
import copy
import sys
sys.setrecursionlimit(10000)


def convert_data(dataset, nclasses=None, part=None):
	""" Convert relation data to LSTM input format """
	dataX = []
	dataY = []
	for tokens, label in dataset:
		if part is not None:
			tokens = tokens[part]
		dataX.append([token2id[token] for token in tokens])
		dataY.append(label2id[label])
	# convert list of lists to array and pad sequences if needed
	X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
	# reshape X to be [samples, time steps, features]
	X = np.reshape(X, (X.shape[0], max_len))
	# one-hot representation
	y = np_utils.to_categorical(dataY, nb_classes=nclasses) # for keras 1
	#y = np_utils.to_categorical(dataY, num_classes=nclasses) # for keras 2
	return X, y


def eval_any(observations, predictions):
	""" Calculate accuracy where match with any of multiple given labels counts as correct """
	corr = 0.
	for obs, pred in zip(observations, predictions):
		if pred in obs:
			corr += 1
	return corr/len(observations)


def shift(X, y, dev_X, dev_y, shuffle=False, val_size=None):
	""" Shift/shuffle train and validation data sets """
	if not val_size:
		val_size = len(dev_X)
		print ("Validation size", val_size)
	X, y = np.concatenate((dev_X,X)), np.concatenate((dev_y,y))
	if shuffle:
		idxs = random.sample(range(len(X)), len(X))
		for i, idx in enumerate(idxs):
			tmpx, tmpy = X[idx], y[idx]
			X[idx], y[idx] = X[i], y[i]
			X[i], y[i] = tmpx, tmpy

	dev_X, dev_y = X[-1*val_size:], y[-1*val_size:]
	X, y = X[:-1*val_size], y[:-1*val_size]
	return X, y, dev_X, dev_y


## Load data

trainset = resources.read_relations("conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-zh-01-08-2016-train/", ignore_types=["Explicit", "AltLex"], partial_sampling=True)
devset = resources.read_relations("conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-zh-01-08-2016-dev/", ignore_types=["Explicit", "AltLex"], partial_sampling=True)
testset = resources.read_relations("conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-zh-01-08-2016-test/", ignore_types=["Explicit", "AltLex"])
"""
trainset = resources.read_relations("conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16-train/", ignore_types=["Explicit", "AltLex"], partial_sampling=True)
devset = resources.read_relations("conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16-dev/", ignore_types=["Explicit", "AltLex"], partial_sampling=True)
testset = resources.read_relations("conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16-test/", ignore_types=["Explicit", "AltLex"])
"""


max_len	= 256	# Maximum input sequence length
# Set maximum input sequence length as percentile of actual lengths
#max_perc = 98.0
#max_len = int(np.percentile([len(smpl[0]) for smpl in trainset+devset+testset], max_perc))
print ("Maximum sequence length", max_len)

vocab = set()
labels = set()
for tokens, label in trainset+devset+testset:
	if type(tokens) is tuple:
		tokens = tokens[0] + tokens[1]
	for token in tokens:
		vocab.add(token)
	labels.add(label)

print ("Vocabulary:", len(vocab))
print ("Classes:", len(labels))

token2id = dict((t, i+1) for i, t in enumerate(vocab))
label2id = dict((t, i) for i, t in enumerate(labels))
id2token = dict((i+1, t) for i, t in enumerate(vocab))
id2label = dict((i, t) for i, t in enumerate(labels))

# Save token and label mapping (is randomly initialized in python3)
json.dump([token2id, label2id, id2token, id2label], open("model.map","w"))

# Read pre-trained word2vec vectors
#vectors = resources.get_vectors(vocab, token2id, "zh-Gigaword-300.txt")
vectors = resources.get_vectors(vocab, token2id, "zh-gw300_intersect.w2v")
emb_dim = vectors.shape[1] # Word embedding dimensions

# Initialize datasets
X, y = convert_data(trainset)
dev_X, dev_y = convert_data(devset, nclasses=y.shape[1])
test_X, test_y = convert_data(testset, nclasses=y.shape[1])

## Define model
batch_size = 80

X, y, dev_X, dev_y = shift(X, y, dev_X, dev_y, val_size=None, shuffle=True)

for nexp in range(5):
	# Repeat experiment
	inlayer1 = Input(shape=(max_len,))
	emb1 = Embedding(len(vocab)+2, emb_dim, input_length=max_len, trainable=True,
					 mask_zero=True, weights=[vectors],
					 dropout=0.5, W_constraint=maxnorm(2)
					)(inlayer1)
	#emb1drop = Dropout(0.5)(inlayer1) # for keras 2: dropout not as argument to emb layer
	lstm1 = Bidirectional(
				LSTM(300, activation="tanh", input_dim=emb_dim, return_sequences=True,
					dropout_W=0.5, W_regularizer=WeightRegularizer(l2=0.0000025)) # for keras 2: W_reg...=l2() instead
			, merge_mode='sum')(emb1)

	attention1 = F.FarATTN(name="M_ATTN1")(lstm1)
	att1drop = Dropout(0.5)(attention1)
	output = Dense(y.shape[1], activation='softmax')(att1drop)

	opt = optimizers.Adam(lr=0.0001)


	## Training and evaluation
	test_senses = resources.read_senses("conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-zh-01-08-2016-test/", ignore_types=["Explicit", "AltLex"])
	#test_senses = resources.read_senses("conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16-test/", ignore_types=["Explicit", "AltLex"])
	test_labels = [[label2id[s] for s in ss] for ss in test_senses]

	model = Model(input=inlayer1, output=output)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


	best_val, best_val_test = 0, 0
	for epoch in range(25):
		_ = model.fit(X, y, nb_epoch=epoch+1, batch_size=batch_size, verbose=1
			#validation_data=(dev_X, dev_y)
			,initial_epoch=epoch
			)
		#scores = model.evaluate(X, y, verbose=1)
		print ("\tdev\ttest")
		dev_scores = model.evaluate(dev_X, dev_y, verbose=0, batch_size=batch_size)
		print ("acc.\t%.2f%%" % (dev_scores[1]*100), end="", flush=True)
		test_scores = model.evaluate(test_X, test_y, verbose=0, batch_size=batch_size)
		print ("\t%.2f%%" % (test_scores[1]*100), end="", flush=True)
		if dev_scores[1] > best_val:
			print ("\t*")
			best_val_test_any = eval_any(test_labels, [np.argmax(y) for y in model.predict(test_X, batch_size=batch_size)])
			best_val = dev_scores[1]
			best_val_test = (best_val_test_any,)
			print ("Test accuracy on any sense:", round(best_val_test_any*100,2))
			#print("Saving model...")
			#model.save("model.keras")
		else:
			print()

	print ("Best validation score:", round(best_val*100,2))
	print ("with test score: %.2f" % tuple([x*100 for x in best_val_test]))

	# Reset model for next experiment
	del inlayer1, emb1, lstm1, attention1, att1drop, output, model, opt


### Build new model to output alpha weights
"""
emb1X = Embedding(len(vocab)+2, emb_dim, input_length=max_len, trainable=True,
				 mask_zero=True, weights=[vectors],
				 dropout=0.5, W_constraint=maxnorm(2)
				 , weights=model.layers[1].get_weights()
				)(inlayer1)
lstm1X = Bidirectional(
			LSTM(300, activation="tanh", input_dim=emb_dim, return_sequences=True
				, dropout=0.5, W_regularizer=WeightRegularizer(l2=0.0000025)
				, weights=model.layers[2].get_weights())
		, merge_mode='concat')(emb1X)
attention1X = F.FarATTN(name="M_ATTN1", weights=model.layers[3].get_weights(), return_alpha=True)(lstm1X)

modelX = Model(input=inlayer1, output=attention1X)

test_alphas = modelX.predict(test_X), batch_size=80)
"""
