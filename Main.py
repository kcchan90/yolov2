import cv2
import numpy as np
import pandas as pd
from detection import *
##For debug only
import matplotlib.pyplot as plt


##fake function (just for testing)
def get_random_boxes():
    no_rect = np.random.randint(1,4)
    boxes = [()]*no_rect
    for i in range(no_rect):
        boxes[i] = [np.random.randint(0,416,4)]
    return boxes,['person']*len(boxes),[1]*len(boxes)

    
###main function
def main(video_name,select_obj,model,anchors,class_names,in_path = 'Input/',out_path='Output/',confid_th = 0.4,overlap_th=0.5):
    if type(select_obj)==str:
        select_obj=[select_obj]
    
    video_path = in_path+video_name
    video_name0 = video_name.split('.')[-2]
    
    video = cv2.VideoCapture(video_path)
    nframe = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ##output config
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') ##mp4 format
    out_video = cv2.VideoWriter(out_path+video_name0+'_output.mp4',fourcc, fps, (416,416))
    out_df = pd.DataFrame(columns=['Frame','Left','Top','Right','Bottom','Object_class'])
    
    ##Config for checking
    show_first_frame=False
    
    frame_no = 0
    while video.isOpened:
        ret,frame = video.read()
        if ret:
            #Resize input
            frame = cv2.resize(frame,(416,416))
            ##copy of image (for model input)
            frame2 = frame.copy()
            frame2 = bgr2rgb(frame2)
            frame2 = frame2.astype(np.float32)/255
            
            ##Convert frame2 to batch data (size=1)
            model_out = model.predict(frame2[None])
            ##Apply model
            boxes, labels, confid_values = read_model_output(model_out[0],anchors,class_names,confid_th=confid_th)
            
            ##For debug only
            #boxes, labels, confid_values = get_random_boxes()
            ##Select only the requested object(s)
            text_labels = np.array(class_names)[labels]
            filter = np.isin(text_labels,select_obj)
            boxes = boxes[filter]
            labels = labels[filter]
            confid_values = confid_values[filter]
            if len(boxes>0):
                if (overlap_th<1) and (len(boxes))>1:
                    boxes, labels, confid_values = box_filter(boxes,labels,confid_values,overlap_th=overlap_th)
                
                text_labels = np.array(class_names)[labels]
                temp_df = pd.DataFrame(boxes,columns = ['Left','Top','Right','Bottom'])
                temp_df['Object_class'] = text_labels
                temp_df['Frame'] = frame_no
                
                out_df = pd.concat([out_df,temp_df])
                ##Drawing
                for i in range(len(boxes)):
                    left,top,right,bottom = boxes[i]
                    frame = cv2.rectangle(frame,(left,top),(right,bottom),(0,100,0),2)
                    frame = cv2.putText(frame,text_labels[i],(left,top-5),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,100,0))
            
            if show_first_frame:
                plt.imshow(frame)
                show_first_frame=False
            out_video.write(frame)
            frame_no+=1
            if frame_no%int(nframe/20)==0:
                print("Finished %d%%"%(frame_no/nframe*100))
        else:
            break

    out_df[['Frame','Left','Top','Right','Bottom','Object_class']].to_csv(out_path+video_name0+'_DF.csv',index=False)
        
    out_video.release()
    video.release()
    ##Checking
    print(out_df[:4])
    return None
        

if __name__ == '__main__':
    class_path = 'model/pascal_classes.txt'
    
    if len(sys.argv)<2:
        video_name  = '1.mp4'
        model_name = 'yolov2-voc'
        print('Using default video (1.mp4) and model (yolov2-vic)')
    elif len(sys.argv)<3:
        video_name  = sys.argv[1]
        model_name = 'yolov2-voc'
        print('Using default model (yolov2-vic)')
    else:
        video_name = sys.argv[1]
        model_name = sys.argv[2]
    
    
    select_obj = pd.read_csv('input/select_obj.txt')['object'].values
    print('Read selectobj.txt: Detecting ' +", ".join(select_obj))
    
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
    
    if model_name =='yolo2-voc':
        ##yolov2-voc took too long time
        main(video_name,select_obj,model,anchors,class_names,confid_th = .7)
    else:
        main(video_name,select_obj,model,anchors,class_names,confid_th = .4)
    
