#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:11:54 2020

@author: vasilispythonlab
"""
import argparse
import tensorfow as tf
import tensorflow_hub as hub
import json
from PIL import Image
import numpy as np
#import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('image_file_name', type=str, default='./cautleya_spicata.jpg',help='Enter file path')
    parser.add_argument('model_file_name', type=str, default='./trained_flower_classif_model.h5',help='Enter model path')
    parser.add_argument('k', type=int, default=5,help='Enter k')
    parser.add_argument('class_name_file', type=str, default='./label_map.json',help='Enter lbels path(json)')
    
    return parser.parse_args()

def load_saved_model(model_name):
    # TODO: Load the Keras model
    trained_model=tf.keras.models.load_model( model_name, custom_objects={'KerasLayer':hub.KerasLayer} )
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


def predict(image_path , model ,class_names , top_k):
    image_name = image_path.split('/')[-1][ : -4]
    print('Reading ....', image_name)
    # read and pre_process image
    im=Image.open(image_path) 
    im=np.array(im)
    im=process_image(im)
    im_final=np.expand_dims(im, axis=0)
  
    #run model.predict to get NN probabilities that image belongs in class j
    prob=model.predict(im_final)[0]
  
    #create dataframe with probability, class_id,class_name
    model_df=pd.DataFrame(prob)
    model_df.rename(columns = {0:'prob'}, inplace = True) 
  
    model_df['class']=model_df.index+1

    first_class_id=str( model_df['class'][0] )
  
    first_flower = class_names.get(first_class_id )
  
    model_df['name']=[ class_names.get(str(i)) for i in model_df['class'] ]
  
  
    model_df.sort_values(by='prob', ascending=False,inplace=True)
  
    return model_df['prob'].head(top_k) , model_df['class'].head(top_k), model_df['name'].head(top_k),im


    
def main():
    args=parse_args()
    print(args)
    
    # read in image file name
    img_path=args.image_file_name
    
    # read in model from the disk
    model=load_saved_model(args.model_file_name)
    
    # read in K
    k=args.k
    
    # read in json file name
    class_names=read_image_mappings_from_json_file(args.class_name_file)
    
    # predict 
    probs,classid,names,img=predict( img_path ,  model  , class_names, k )

    # output
    predicted_class_name=names[0]
    print('\n Image File Name:{} , Predicted Class:{}'.format(img_path,predicted_class_name))
    plt.imshow(img)
    plt.title(predicted_class_name)
    
if __name__=='__main__':
    main()
    
    
