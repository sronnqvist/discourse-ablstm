"""
LSTM classifier for discourse relation senses
author: Samuel Ronnqvist <sronnqvi@abo.fi>
source: https://github.com/sronnqvist/discourse-ablstm
"""

import numpy as np
import json
from FarrokhAttentionLayer import FarATTN
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import resources


def convert_data(dataset, nclasses=None, part=None):
	"""" Convert relation data to LSTM input format """
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


## Load data

trainset = resources.read_relations("conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-zh-01-08-2016-train/", ignore_types=["Explicit", "AltLex"])
devset = resources.read_relations("conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-zh-01-08-2016-dev/", ignore_types=["Explicit", "AltLex"])
testset = resources.read_relations("conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-zh-01-08-2016-test/", ignore_types=["Explicit", "AltLex"])
max_len	= 256	# Maximum input sequence length

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

token2id, label2id, id2token, id2label = json.load(open("repo/discourse-ablstm/model7301.map"))

# Read pre-trained word2vec vectors
#vectors = resources.get_vectors(vocab, token2id, "zh-Gigaword-300.txt")
vectors = resources.get_vectors(vocab, token2id, "zh-gw300_intersect.w2v")
emb_dim = vectors.shape[1] # Word embedding dimensions

# Initialize dataset
test_X, test_y = convert_data(testset, nclasses=len(labels))

# Load model
batch_size = 80
print("Loading model...")
model = load_model('repo/discourse-ablstm/model7301.keras', {'FarATTN': FarATTN})
print("Predicting...")
preds = model.predict(test_X, batch_size=batch_size)
print(preds)

# Evaluate
# model.evaluate(test_X, test_y, batch_size=batch_size)
