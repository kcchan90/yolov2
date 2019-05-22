import cv2, keras
import tensorflow as tf
import numpy as np
import pandas as pd


if __name__ == '__main__':
    ## Read video 
    Video = cv2.VideoCapture('test.mp4')
    ## preprocess video
    Video = clean_video(Video)
    
    ## Main function
    Video_edited,data = Main(Video)
    
    
    ##write output
    data.to_csv('output.csv',index=False)
    out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    while True:
        ret,frame = Video_edited.read()
        if ret:
            out.write(frame)
        else:
            break
    out.release()
    cv2.destroyAllWindows() 
    
