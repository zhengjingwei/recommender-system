# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:37:06 2018

@author: kevinshuang
"""
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


model_name="1518095557"
model_name_meta="model"+model_name+".meta"
model_path = "./"+model_name
model_file = "./"+model_name+"/model01"
        
# Create a builder
builder = tf.saved_model.builder.SavedModelBuilder(model_path)
tf.reset_default_graph()  
new_graph = tf.Graph()  
# Add graph and variables to builder and save
with tf.Session(graph=new_graph) as restore_sess:
    restore_saver = tf.train.import_meta_graph(model_name_meta)
    restore_saver.restore(restore_sess,tf.train.latest_checkpoint('./'))
    
    feat_value = new_graph.get_tensor_by_name("feat_value:0")
    out = new_graph.get_tensor_by_name("out:0")

   
    builder.add_meta_graph_and_variables(
          restore_sess,
          [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              "deepfm_model": tf.saved_model.signature_def_utils.predict_signature_def(
                  # inputs is feature vector
                  inputs={"features": feat_value },
                  # outputs is score
                  outputs={"results": out})
          })
              
    builder.save(as_text=1)  