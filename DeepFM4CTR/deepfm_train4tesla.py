# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:02:54 2018

@author: kevinshuang
"""
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from tensorflow.contrib.layers import create_feature_spec_for_parsing
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
import random  
from time import time
import argparse

TIME=int(time())


###tensorboard --logdir="./tmp/deepfm"

def create_deep_fm(inputs, feature_size ,MODETYPE=0, deep_layers=[64,64],dropout_keep_deep=[0.5,0.5,0.5],dropout_keep_fm=[1,1]):
    """
      Args:
          inputs: the input data 
          feature_size: the number of feature size
          deep_layers: the deep struct
          dropout_keep_deep: set dropout for deep part 
          dropout_keep_fm: set dropout for fm part          
      return:
          preidict score
    """
    
    tf.set_random_seed(int(time()))
    
    with tf.name_scope("fm"):
       
        embedding_size=10        
        # embeddings
        embeddings = tf.Variable(
            tf.random_normal([feature_size,embedding_size], 0.0, 0.01),
            name="feature_embeddings")

        # model
        #self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],self.feat_index)
        # None * F * Ks
        
        feat_value = tf.reshape(inputs, shape=[-1, feature_size, 1])

        # ---------- first order term ----------
        ##初始化wij
        feature_bias = tf.Variable(tf.random_uniform([feature_size, 1], 0.0, 1.0), name="feature_bias_0")  # feature_size * 1
        y_first_order = feature_bias
        ## wij * xij 
        y_first_order = tf.reduce_sum(tf.multiply(y_first_order, feat_value), 2)  # None * F
        ## 增加dropout，防止过拟合
        y_first_order = tf.nn.dropout(y_first_order, dropout_keep_fm[0]) # None * F

        # ---------- second order term ---------------
        #vi*xi
        embeddings = tf.multiply(embeddings, feat_value)
        # 和平方
        summed_features_emb = tf.reduce_sum(embeddings, 1)  # None * K
        summed_features_emb_square = tf.square(summed_features_emb)  # None * K

        # 平方和
        squared_features_emb = tf.square(embeddings)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K

        # 和平方与平方和按公式组合
        y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
        y_second_order = tf.nn.dropout(y_second_order, dropout_keep_fm[1])  # None * K
    
    with tf.name_scope("deep"):
        
        # ---------- Deep component ----------
        y_deep = tf.reshape(embeddings, shape=[-1, feature_size*embedding_size]) # None * (F*K)
        y_deep = tf.nn.dropout(y_deep, dropout_keep_deep[0])
        
        weights = dict()
        ##初始化各层的权重
        input_size = feature_size * embedding_size
        glorot = np.sqrt(2.0 / (input_size + deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, deep_layers[0])), dtype=np.float32, name="weights_layer0")
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[0])),
                                                        dtype=np.float32,name="weights_bias0")

        num_layer = len(deep_layers)
        for i in range(1, num_layer): 
            glorot = np.sqrt(2.0 / (deep_layers[i-1] + deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(deep_layers[i-1], deep_layers[i])),
                dtype=np.float32 ,name="weights_layer"+str(i))  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[i])),
                dtype=np.float32 ,name="weights_bias"+str(i))  # 1 * layer[i]
        ##对dnn的各层进行连接        
        for i in range(0, len(deep_layers)):           
            y_deep = tf.add(tf.matmul(y_deep, weights["layer_%d" %i]), weights["bias_%d"%i]) # None * layer[i] * 1
                #if self.batch_norm:
                #    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
            y_deep = tf.nn.relu(y_deep)
            y_deep = tf.nn.dropout(y_deep, dropout_keep_deep[1+i]) # dropout at each Deep layer

    # ---------- DeepFM ----------
    with tf.name_scope("deepfm"):
        concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
        if MODETYPE==0:##deepfm
            concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
            input_size = feature_size + embedding_size + deep_layers[-1]
        elif MODETYPE==1:##fm only
            concat_input = tf.concat([y_first_order, y_second_order], axis=1)
            input_size = feature_size + embedding_size 
        elif MODETYPE==2:##dnn only
            concat_input = y_deep   
            input_size =  deep_layers[-1]
        
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32 ,name="concat_projection0")  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32 ,name="concat_bias0")    
        out = tf.add(tf.matmul(concat_input, weights["concat_projection"]), weights["concat_bias"],name='out')

    score=tf.nn.sigmoid(out,name='score')
    ##观看变量
    tf.summary.histogram("deep+fm"+"/score",score) 
    
    return score

def placeholder_inputs(feature_num, class_num):
    """
      Args:
          feature_num: the number of feature size
          class_num: the number of class
      return:
             two placeholder
    """
    feature_placeholder = tf.placeholder(tf.float32, shape=[None, feature_num],name="input")
    label_placeholder = tf.placeholder(tf.float32, shape=[None, class_num])
    return feature_placeholder, label_placeholder


def save_tf_serving_model(model_dir, sess, feature_placeholder, prediction):
    """
    to save model for tensorflow serving
    :param model_dir: model dir
    :param sess:
    :param feature_placeholder:
    :param prediction:
    :return:
    """
    # 按时间戳保存模型文件，提供给tesorflow serving
    model_dir = model_dir + "/" + str(TIME)
    builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
    builder.add_meta_graph_and_variables(
          sess,
          [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              "serving_default": tf.saved_model.signature_def_utils.predict_signature_def(
                  # inputs is feature vector
                  inputs={"inputs": feature_placeholder},
                  # outputs is score
                  outputs={"results": prediction})
          })
    builder.save(as_text=True) 
                
                
def train(args):                
    ##read file    
    ids = [str(i) for i in range(0,10)]
    feature = [str(i) for i in range(1,10)]
        
    all_df = pd.read_table(args.log_dir,header=None,encoding='utf8',sep=',',skiprows=1,names=ids)
    all_df = all_df.sample(frac = 1)
    
    feature_df = all_df[feature]
    minmax_scala=preprocessing.MinMaxScaler(feature_range=(0,1))
    scalafeature=minmax_scala.fit_transform(feature_df)
    scalafeature_frame = pd.DataFrame(scalafeature ,columns=feature_df.columns) 
    
    ##reset graph
    tf.reset_default_graph()
    y_data = np.array(all_df['0']).reshape(-1,1)
    x_data = np.array(scalafeature_frame).reshape(-1,len(feature)) 
    
    ##get input placeholder
    x, y = placeholder_inputs(len(feature), 1)
    
    ###design deepfm####
    prediction=create_deep_fm(x,len(feature),args.model_type)
    
    if args.model_type==0:
        estimate_name="DeepFm_Estimate"
    elif args.model_type==1:
        estimate_name="Fm_Estimate"
    elif args.model_type==2:
        estimate_name="Deep_Estimate"
        
    with tf.name_scope(estimate_name):
        # 损失函数的定义：均方差
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction),reduction_indices=[1]))
        ##观看常量
        tf.summary.scalar('loss',loss)
        
        auc = tf.contrib.metrics.streaming_auc(prediction,tf.convert_to_tensor(y))   
        ##观看常量
        tf.summary.scalar('auc1',auc[0])
        tf.summary.scalar('auc2',auc[1])
    
    
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(loss)
        init = tf.global_variables_initializer()
        
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        #合并到Summary中    
        merged = tf.summary.merge_all()    
        #选定可视化存储目录  
        writer = tf.summary.FileWriter('./tmp/deepfm',graph=tf.get_default_graph())
        
        
        for _ in range(400):
            sess.run(train_step, feed_dict={x: x_data, y: y_data})
            if _ % 5 == 0:
                #print(str(_)+":loss=")
                print("[%d]loss:%s"%(_,sess.run(loss, feed_dict={x: x_data, y: y_data})))
                result = sess.run(merged,feed_dict={x: x_data, y: y_data}) #merged也是需要run的    
                writer.add_summary(result,_) #result是summary类型的，需要放入writer中，i步数（x轴）
                #auc=sess.run(auc, feed_dict={x: x_data, y: y_data})
                #print(auc[0])
                #print(sess.run(prediction, feed_dict={x: x_data, y: y_data}))
                #print(sess.run(prediction, feed_dict={x: x_data, y: y_data}).shape)
                #print(y.shape)
        
        print("test:")       
        test_x=x_data[0].reshape(-1,9)
        print(test_x)
        print(sess.run(prediction, feed_dict={x: test_x}))
    
        save_tf_serving_model("./save", sess, x, prediction)
        meta_file="./save/model"+str(TIME)
        saver.save(sess,meta_file)
        tf.train.write_graph(sess.graph_def, "./save", "test.pb", False)
        
        writer.close()
    sess.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--log_dir',
      type=str,
      default='./data/test.txt',
      help='Directory to put the log data.'
    )   
    parser.add_argument(
      '--model_type',
      type=int,
      default=2,
      help='Directory to put the log data.'
    )
    """
       MODETYPE: 0 deepfm ,1 only fm ,2 only deep
    """       
    args = parser.parse_args()
    print(args.log_dir)
    print(args.model_type)
    train(args)

 