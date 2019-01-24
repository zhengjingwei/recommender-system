# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:37:06 2018

@author: kevinshuang
"""
import math 
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

model_path = "./save/"+str(int(time())) 
model_file = "./save/"+str(int(time()))+"/model01"
        
#tf.reset_default_graph()

# Add graph and variables to builder and save
tf.reset_default_graph()
restore_graph = tf.Graph()

with tf.Session(graph=restore_graph) as restore_sess: 
    restore_saver = tf.train.import_meta_graph('model1518145672.meta')
    restore_saver.restore(restore_sess,'./model1518145672')
    feat_value = tf.get_default_graph().get_tensor_by_name("input:0")
    out = tf.get_default_graph().get_tensor_by_name("score:0")
    Xv_train=np.array([[0.53216693 , 0.69765617 , 0.0990991 ,  0.0204531 ,  0.01580161,  0.02709669
  , 0.0071048 ,  0.01450924 , 0.00912575]])
    Xv_train=Xv_train.reshape((-1,9))
    print(Xv_train.shape)
    ans = restore_sess.run(out, feed_dict={feat_value:Xv_train})
    print(ans)