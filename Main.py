import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from function import get_human


##fake function (just for testing)
def get_human(img,frame_no):
    no_rect = np.random.randint(1,4)
    list_rect = [()]*no_rect
    for i in range(no_rect):
        list_rect[i] = (np.random.randint(0,416,2),np.random.randint(0,416,2))
    rect_df = pd.DataFrame(list_rect,columns= ['LD','RU'])
    rect_df['Frame']=frame_no
    return rect_df

##example
video_name = 'test_video/1.mp4'
#video_name = 'test_video/2.avi'

dot_pos = len(video_name) - video_name[::-1].find('.')
video_name0 = video_name[:dot_pos-1]
video = cv2.VideoCapture(video_name)
nframe = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

##output config
fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
size = (416,416)
out = cv2.VideoWriter(video_name0+'_edited.mp4',fourcc, fps, size)
out_df = pd.DataFrame(columns=['Frame','LD','RU'])

##for checking
show_first_frame=True

##plt: rgb
##cv2: bgr
frame_no=0
while video.isOpened:
    ret,frame = video.read()
    if ret:
        #Resize input
        frame2 = cv2.resize(frame,(416,416))
        ##apply Yolov2
        rect_df = get_human(frame2,frame_no)
        out_df = pd.concat([out_df,rect_df])
        #draw rectange (Left down,Right up)
        for idx,row in rect_df.iterrows():
            frame2 = cv2.rectangle(frame2,tuple(row['LD']),tuple(row['RU']),(0,0,255),3)
        if show_first_frame:
            plt.imshow(frame2)
            show_first_frame=False
        out.write(frame2)
        frame_no+=1
        #print(frame2.shape)
    else:
        break

        
#out_df[['Frame','LD','RU']].to_csv('output/'+video_name0+'.csv',index=False)
out_df[['Frame','LD','RU']].to_csv(video_name0+'_DF.csv',index=False)
out.release()
video.release()

