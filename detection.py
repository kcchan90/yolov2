## functions for detecting object
##Reference: https://github.com/guigzzz/Keras-Yolo-v2

from keras.models import load_model
import cv2, configparser
import numpy as np
import sys

##For debug only
#import matplotlib.pyplot as plt

####################################################################
#######################Function from Keras-Yolo-v2##################
def logistic(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_out = np.exp(x - np.max(x, axis=-1)[..., None])
    return exp_out / np.sum(exp_out, axis=-1)[..., None]
####################################################################


def bgr2rgb(img):
    b,g,r = cv2.split(img)
    return cv2.merge([r,g,b])

    
##Function to read model output for one image and return boxes,labels,and confidence values
def read_model_output(MO,anchors,class_names,confid_th=0.5,cell_size = 416/13):
    ##MO: model output
    ##Reshape model output to (n_row_cell,n_col_cell,n_anchors,(4 boxes parameters + 1 box confidence + n_class))
    MO = MO.reshape(13,13,len(anchors),5+len(class_names))
    ## boxes' parameters before transformation
    temp_x = MO[..., 0]
    temp_y = MO[..., 1]
    temp_w = MO[..., 2]
    temp_h = MO[..., 3]
    temp_c = MO[..., 4]
    
    ## boxes' parameters after transformation
    pw,ph = anchors[:, 0], anchors[:, 1]
    ## box_x, box_y are also adjusted for cell location
    box_x = logistic(temp_x) + np.arange(13)[None,:,None]
    box_y = logistic(temp_y) + np.arange(13)[:,None,None]
    box_w = pw * np.exp(temp_w)/2
    box_h = ph * np.exp(temp_h)/2
    box_confid = logistic(temp_c)
    ## for drawing rectangles
    left   = (box_x-box_w) * cell_size
    right  = (box_x+box_w) * cell_size
    top    = (box_y-box_h) * cell_size
    bottom = (box_y+box_h) * cell_size
    
    ##list of potential objects
    boxes = np.stack((left,top,right,bottom),axis=-1)
    
    ##softmax transform for classes' prediction
    softmax_values = softmax(MO[...,5:])
    ##Best prediction for each anchor
    labels = np.argmax(softmax_values, axis=-1)
    class_confid = np.max(softmax_values, axis=-1)
    ## Final confidence
    confid_values = class_confid * box_confid
    ## Filter boxes by confidence threshold
    boxes  = boxes[confid_values>confid_th].astype(np.int32)
    labels = labels[confid_values>confid_th]
    confid_values = confid_values[confid_values>confid_th]
    
    return boxes,labels,confid_values

##Drop redundant boxes with high overlapping area:   
def box_filter(boxes,labels,confid_values,overlap_th):
    indx = np.argsort(-confid_values)
    boxes = boxes[indx]
    confid_values = confid_values[indx]
    labels = labels[indx]
    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]
    area = (x2 - x1+1) * (y2 - y1+1)
    pick = np.ones(len(labels)).astype(bool)
    for i in range(len(labels)-1):
        if pick[i]:
            temp_x1 = np.maximum(x1[i], x1[(i+1):])
            temp_y1 = np.maximum(y1[i], y1[(i+1):])
            temp_x2 = np.minimum(x2[i], x2[(i+1):])
            temp_y2 = np.minimum(y2[i], y2[(i+1):])
            
            w = np.maximum(0, temp_x2 - temp_x1 + 1)
            h = np.maximum(0, temp_y2 - temp_y1 + 1)
            
            ##intersection area / union area
            Check = (w*h/(area[i]+area[(i+1):]-w*h))>overlap_th
            pick[(i+1):][Check]=False
            
    return boxes[pick],labels[pick],confid_values[pick]
    
    
def detect_one_imgage(img_name,model,anchors,class_names,in_path = 'input/',out_path = 'output/',confid_th = 0.5,overlap_th=.5):
    img = cv2.imread(in_path+img_name)
    img_h,img_w,img_ch = img.shape
    img = cv2.resize(img,(416,416))
    
    ##copy of image (for model input)
    img2 = img.copy()
    img2 = bgr2rgb(img)
    img2 = img2.astype(np.float32)/255
    ###
    ##Convert img2 to batch data (size=1)
    model_out = model.predict(img2[None])
    boxes, labels, confid_values = read_model_output(model_out[0],anchors,class_names,confid_th=confid_th)
    ##Drop redundant boxes
    if (overlap_th<1) and (len(boxes))>1:
        boxes, labels, confid_values = box_filter(boxes,labels,confid_values,overlap_th=overlap_th)
    
    text_labels = np.array(class_names)[labels]
    for i in range(len(boxes)):
        left,top,right,bottom = boxes[i]
        img = cv2.rectangle(img,(left,top),(right,bottom),(0,100,0),2)
        img = cv2.putText(img,text_labels[i],(left,top-5),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,100,0))
        
    img = cv2.resize(img,(img_w,img_h))
    #plt.imshow(img) ##for debug
    img_name0 = img_name.split('.')[-2]
    cv2.imwrite(out_path+img_name0+'_output.jpg',img)
    
    return None
    
    
    
if __name__ == '__main__':
    ###An Example for using the functions
    class_path = 'model/pascal_classes.txt'
    ##default setting
    if len(sys.argv)<2:
        img_name  = 'test1.jpg'
        model_name = 'yolov2-voc'
        print('Using default image and model (yolov2-vic)')
    elif len(sys.argv)<3:
        img_name  = sys.argv[1]
        model_name = 'yolov2-voc'
        print('Using default model (yolov2-vic)')
    else:
        img_name = sys.argv[1]
        model_name = sys.argv[2]
    
    ##read class names
    with open(class_path,'r') as wf:
        class_names = wf.readlines()
        class_names = [c.strip() for c in class_names]

    ##read anchor config
    with open('model/'+model_name+'_anchors.txt','r') as wf:
        anchors = wf.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
        
    model = load_model('model/'+model_name+'.h5')   
     
    model_input_shape = model.layers[0].input_shape[1:3] 
    model_output_size = model.layers[-1].output_shape[-1]
    ##Checking
    if (model_output_size != len(anchors)*(len(class_names)+5)) or (model_input_shape != (416,416)):
        raise ValueError('Input Shape not match!!!Check Config or Class file')

    detect_one_imgage(img_name,model,anchors,class_names)


     
