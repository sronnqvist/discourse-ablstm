# Att-BiLSTM neural network for discourse sense classification
Attention-based Bidirectional Long Short-Term Memory model for classification of Chinese implicit discourse relations.

Code developed by Samuel Rönnqvist, with contributions and comments from Farrokh Mehryary, Niko Schenk and Philip Schultz.

This repository hosts the model described in: 

Samuel Rönnqvist, Niko Schenk and Christian Chiarcos. [A Recurrent Neural Model with Attention for the Recognition of Chinese Implicit Discourse Relations](https://arxiv.org/pdf/1704.08092.pdf). Forthcoming in *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)*. 2017.

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

The work is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), for academic use please cite above paper.

## Requirements

Installing required software:

```
pip for python3, e.g.:
apt-get install python3-pip

pip3 install numpy h5py gensim
pip3 install git+git://github.com/fchollet/keras.git --upgrade --no-deps
```

Set Keras backend to "theano" in ~/.keras/keras.json!

Note! Code is developed for Keras 1.2.x, not fully compatible with Keras 2.

For training model, run:
`python3 train.py`

Data is available through the [CoNLL-2016 Shared Task](http://www.cs.brandeis.edu/~clp/conll16st/dataset.html).




