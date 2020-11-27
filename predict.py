#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:11:54 2020

@author: vasilispythonlab
"""
import matplotlib.pyplot as plt
#import pandas as pd
import numpy as np

import argparse
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
#import matplotlib as plt


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('image_file_name', type=str, default='./test_images/cautleya_spicata.jpg',help='Enter file path')
    parser.add_argument('model_file_name', type=str, default='./trained_flower_classif_model.h5',help='Enter model path')
    parser.add_argument('k', type=int, default=5,help='Enter k')
    parser.add_argument('class_name_file', type=str, default='./label_map.json',help='Enter lbels path(json)')
    
    return parser.parse_args()


def load_saved_model(model_name):
    # TODO: Load the Keras model
    trained_model=tf.keras.models.load_model( model_name, custom_objects={'KerasLayer':hub.KerasLayer} , compile=False)
    return trained_model


def read_image_mappings_from_json_file(json_file):
    with open( json_file, 'r') as f:
        class_names = json.load(f)
    return class_names

def process_image(in_image):
  out_image=np.squeeze(in_image)
  out_image=tf.image.resize( out_image , (224,224) )
  out_image= out_image / 225
  return out_image


def predict_without_pandas(image_path , model ,class_names , top_k):
  image_name = image_path.split('/')[-1][ : -4]
  print('Reading ....', image_name)
  # read and pre_process image
  im=Image.open(image_path) 
  im=np.array(im)
  im=process_image(im)
  im_final=np.expand_dims(im, axis=0)
  
  #run model.predict to get NN probabilities that image belongs in class j
  prob=model.predict(im_final)[0]
  prob_list=prob.tolist()
    
  class_list=list(range(len(prob_list)))
  class_list=[ i+1 for i in class_list ]
  
  #prob_class_list=[(prob_list[i], class_list[i]) for i in range(0, len(prob_list))] 
  prob_class_list= tuple( zip( prob_list , class_list ) )
  sort_prob_list=sorted(prob_list, reverse=True )
  
  sorted_prob_class = [ tuple for x in sort_prob_list for tuple in prob_class_list if tuple[0] == x ] 
  print( sorted_prob_class[ : top_k] )

  unzipped_sorted_prob_class_topk = list( zip( *sorted_prob_class[ : top_k]))
  prob_ret=unzipped_sorted_prob_class_topk[0]
  class_ret=unzipped_sorted_prob_class_topk[1]
  name_ret=[ class_names.get( str(i) ) for i in class_ret ]
  
  return prob_ret , class_ret , name_ret , im

    
def main():
    args=parse_args()
    print('Here are the args:' , args)
    
    # read in image file name
    img_path=args.image_file_name
    
    # read in model from the disk
    model=load_saved_model(args.model_file_name)
    
    # read in K
    k=args.k
    
    # read in json file name
    class_names=read_image_mappings_from_json_file(args.class_name_file)
    
    # predict 
    probs,classid,names,img=predict_without_pandas( img_path ,  model  , class_names, k )

    # output
    predicted_class_name=names[0]
    print('\n Image File Name:{} , Predicted Class:{}'.format(img_path,predicted_class_name))
    #plt.imshow(img)
    #plt.title(predicted_class_name)
    
if __name__=='__main__':
    main()
    
    
