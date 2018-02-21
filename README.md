# Att-BiLSTM neural network for discourse sense classification
Attention-based Bidirectional Long Short-Term Memory model for classification of Chinese implicit discourse relations.

Code developed by Samuel Rönnqvist, with contributions and comments from Farrokh Mehryary, Niko Schenk and Philip Schultz.

**NOTE**: The original model from these authors **performs bad** on the blind test dataset from CoNLL2016st! It also **cheats by shuffling** the training and validation datasets and show much better validation score! The partial sampling implementation differs from the one in the paper, because it contains a **bug**. Otherwise, the **attention layer is unusual**, because it is computed in respect to a fixed query vector.

Results after fixing these issues (see `output-corrected.txt` and `output-original.txt`):

```
Trainable params: 6,569,710
Best validation score: 64.78
with test score: 69.03
with blind score: 62.99
```

This repository hosts the corrected model described in: 

Samuel Rönnqvist, Niko Schenk and Christian Chiarcos. [A Recurrent Neural Model with Attention for the Recognition of Chinese Implicit Discourse Relations](https://arxiv.org/pdf/1704.08092.pdf). In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)*. 2017.

```
@inproceedings{ronnqvist2017ablstm,
  author    = {Samuel R\"onnqvist and Niko Schenk and Christian Chiarcos},
  title     = {{A Recurrent Neural Model with Attention for the Recognition of Chinese Implicit Discourse Relations}},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver, BC, Canada},
  publisher = {Association for Computational Linguistics}
}
```

The paper is also available as a [poster](https://github.com/sronnqvist/discourse-ablstm/blob/master/acl_poster.pdf), which was presented at the ACL conference.

The work is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), for academic use please cite above paper.

## Requirements

Installing required software:

```
apt-get install python3 python3-pip

pip3 install numpy h5py gensim
pip3 install keras==1.2.0

gzip -d zh-gw300_intersect.w2v.gz
```

For training use:

```
KERAS_BACKEND=theano python3 train.py
```

Code is developed for Keras 1.2.x, not fully compatible with Keras 2. Use Keras backend "theano".

Alternatively run it inside the Docker container [gw000/keras-full](https://hub.docker.com/r/gw000/keras-full/):

```
docker run -it --rm -v $(pwd):/srv --user root gw000/keras-full:1.2.0 bash
pip3 install numpy h5py gensim
KERAS_BACKEND=theano python3 train.py
```

For training model on GPU using Docker:

```
docker run -it --rm $(ls /dev/nvidia* | xargs -I{} echo '--device={}') $(ls /usr/lib/*-linux-gnu/{libcuda,libnvidia}* | xargs -I{} echo '-v {}:{}:ro') -v $(pwd):/srv --user root gw000/keras-full:1.2.0 bash
pip3 install numpy h5py gensim
KERAS_BACKEND=theano THEANO_FLAGS='device=gpu,floatX=float32,nvcc.fastmath=True,lib.cnmem=0.45' python3 train.py
```

Data is provided through LDC and was used for [CoNLL-2016 Shared Task](http://www.cs.brandeis.edu/~clp/conll16st/dataset.html).

