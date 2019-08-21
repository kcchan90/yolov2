# Convert Config and Weights to keras format
# Reference: https://github.com/allanzelener/YAD2K
###Update: darknet weights file format changed,
##See: https://github.com/allanzelener/YAD2K/issues/65#issuecomment-325184900
##Weights Header read(16)->read(20) for yolov2-voc but read(16) for yolov2-tiny-voc


import configparser, sys
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import (Conv2D, Input, Lambda, MaxPooling2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


########################################################################################
###############Function from YAD2K######################################################
def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.
    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])
#######################################################################################

#rewrite yolov2-voc.cfg to rename duplicate section names
def rewrite_cfg(model_name):
    sect_count={}
    with open('model/'+model_name+'.cfg','r') as rf:
        with open('model/'+model_name+'_edited.cfg','w') as wf:
            for row in rf:
                if row[0]=='[':
                    sect = row.strip('[]\n')
                    try:
                        sect_count[sect]+=1
                    except:
                        sect_count[sect]=0
                    sect += '_'+str(sect_count[sect])
                    row = '[%s]\n'%sect
                wf.write(row)
                
                
def build_yolo(model_name):
    Config = configparser.ConfigParser()
    Config.read('model/'+model_name+'_edited.cfg')
    dknet_weights = open('model/'+model_name+'.weights', 'rb')
    ##weight header
    if model_name == 'yolov2-voc':
        weights_header = np.ndarray(shape=(4, ), dtype='int32', buffer=dknet_weights.read(20))
    else:
        weights_header = np.ndarray(shape=(4, ), dtype='int32', buffer=dknet_weights.read(16))
    # print('Weights Header: ', weights_header)
    
    #input size
    img_h, img_w = int(Config['net_0']['height']), int(Config['net_0']['width'])
    
    ##Weight
    ##Start build the model
    prev_layer = Input(shape = (img_h,img_w,3))
    all_layers = [prev_layer]
    weight_decay = float(Config['net_0']['decay'])
    
    layer_count = 0
    for section in Config.sections():
        print('Working for section: %s'%section)
        if section.find('net_')>-1:
            ##network config
            pass
        elif section.find('convolutional_')>-1:
            ##Convolution Layer
            ##weights input form: [bias/beta, [gamma, mean, variance], conv_weights]
            filters, size, stride, pad, activation = int(Config[section]['filters']), int(Config[section]['size']),\
                                                        int(Config[section]['stride']),int(Config[section]['pad']),Config[section]['activation']
            bn = 'batch_normalize' in Config[section]
            padding = 'same' if pad == 1 else 'valid'
            
            prev_layer_shape = K.int_shape(prev_layer)   #input shape
            
            darknet_w_shape = (filters, prev_layer_shape[-1], size, size) #darknet weight shape
            weights_shape = (size, size, prev_layer_shape[-1], filters)   #keras weight shape
            weights_size = np.product(darknet_w_shape)    #total number of weights in this layer
            
            # Print layer config
            layer_count+=1
            print('Layer %d: Conv2d with%s bn\n\t Activation: %s\n\t Weights_shape: '%(layer_count,'' if bn else 'out',activation),weights_shape)
            
            ##Load parameters for bias, batch normalization and weights
            #Bias
            conv_bias = np.ndarray(shape=(filters,), dtype='float32', buffer =dknet_weights.read(filters*4))
            
            #Batch Normalization
            if bn:
                bn_weights = np.ndarray(shape=(3, filters), dtype='float32', buffer =dknet_weights.read(filters*12))
                bn_weight_list  = [bn_weights[0],conv_bias,bn_weights[1],bn_weights[2]] ## [scale gamma,shift beta, mean, var]
            
            #Convolution Weights
            conv_weights = np.ndarray(shape=darknet_w_shape, dtype='float32', buffer =dknet_weights.read(weights_size*4))
            #need to convert darknet weight shape to keras weight shape
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights  = [conv_weights] if bn else [conv_weights,conv_bias]
            
            #Activation function
            if activation == 'leaky':
                act_func = None
            elif activation == 'linear':
                act_func = 'linear'
            else:
                raise ValueError('Unknown activation in layer %d : %s'%(layer_count,activation))
                
            #Build layer
            conv_layer = (Conv2D(filters,(size,size),strides = (stride, stride),kernel_regularizer = l2(weight_decay),\
                            use_bias = not bn,weights=conv_weights,activation = act_func,padding=padding))(prev_layer)
                            
            if bn:
                conv_layer = (BatchNormalization(weights = bn_weight_list))(conv_layer)
            
            prev_layer = conv_layer
            if activation == 'linear':
                all_layers.append(prev_layer)
            else:
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
                
        elif section.find('maxpool_')>-1:
            #MaxPooling layer (No parameters to load)
            layer_count+=1
            print('Layer %d: MaxPool'%layer_count)
            
            size = int(Config[section]['size'])
            stride = int(Config[section]['stride'])
            pool_layer = (MaxPooling2D(padding = 'same',pool_size = (size,size),strides = (stride,stride)))(prev_layer)
            prev_layer = pool_layer
            all_layers.append(prev_layer)
        
        elif section.find('route_')>-1:
            ##Route layer
            ## for concatenating or loading previous layer
            ids = [int(i) for i in Config[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids] # load previous layers
            if len(ids)==1:
                layer_count+=1
                print('Layer %d: Loading Layer %d'%(layer_count,layer_count-ids[0]))
                skip_layer = layers[0]
                all_layers.append(skip_layer)
                prev_layer = skip_layer
            else:
                layer_count+=1
                print('Layer %d: Concatenating Layer '%(layer_count),[layer_count+i for i in ids])
                concatenate_layer = concatenate(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
                
        elif section.find('reorg_')>-1:
            ##Reshape tensor from (s1,s2,s3,s4) to (s0,s1/2,s2/2,s3*4)
            if int(Config[section]['stride'])!=2:
                raise ValueError('Stride for reorg can only be 2')
                
            layer_count+=1
            print('Layer %d: Reshape')
            reorg_layer = (Lambda(space_to_depth_x2, output_shape=space_to_depth_x2_output_shape,name='space_to_depth_x2'))(prev_layer)
            all_layers.append(reorg_layer)
            prev_layer = reorg_layer
        
        elif section.find('region_')>-1:
            ##write anchor config
            with open('model/'+model_name+'_anchors.txt', 'w') as wf:
                wf.write(Config[section]['anchors'])
        else:
            raise ValueError('Cannot handle section %s'%section)
    
    ##Create and save model object
    model = Model(inputs=all_layers[0], outputs=all_layers[-1])
    print(model.summary())
    model.save('model/'+model_name+'.h5')
    print("model "+model_name+' saved')
    ## For checking
    remaining_weights = len(dknet_weights.read()) / 4
    dknet_weights.close()
    if remaining_weights==0:
        print('ALL weights loaded, no remaining weights. Success :) ')
    else:
        print('SOME weights remaining, there should be some bugs!!!!!',remaining_weights)
    
    return remaining_weights

if __name__ =='__main__':       
    if len(sys.argv)<2:
        raise ValueError('Please input model name: yolov2-voc or yolov2-tiny-voc')
    else:
        #model_name = 'yolov2-voc'
        #model_name = 'yolov2-tiny-voc'
        model_name = sys.argv[1]
        
    rewrite_cfg(model_name)
    build_yolo(model_name)

    
    
