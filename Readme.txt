For YOLOv2 testing

Required Packages:
tensorflow==1.3.0
keras==2.1.2
cv2==4.1.0

Remarks: latest version of keras may have bugs for keras.layers.advanced_activations.LeakyReLU
Link:https://github.com/keras-team/keras/issues/9349



How to Use:
1. Download weights and cfg from here to model/ (only support yolov2-voc and yolov2-tiny-voc)

2. Convert darknet weights and cfg to keras format
e.g.
python convert_model yolov2-voc
python convert_model yolov2-tiny-voc

3a). Detect Object in an image located at input/
e.g.
python detection yolov2-tiny-voc test1.jpg
python detection yolov2-voc test2.jpg

3b). Detect Object in a video located at input/ (edit input/select_obj.txt to detect more types of objects)
e.g.
python Main yolov2-voc 1.mp4
python Main yolov2-tiny-voc 2.avi

4. ALL output files (image/video/csv) are stored in output/



Files description:
- convert_model.py:
    Convert darknet cfg and weights files to readable file for python keras.

- detection.py:
    Functions for reading model output and example for detecting objects in pictures
    
- Main.py:
    Main functions for detecting people in videos
    
Remarks:
You can also detect other labels mentioned in model/pascal_class.txt by editing input/select_obj.txt
