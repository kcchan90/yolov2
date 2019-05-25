# yolov2

For YOLOv2 testing

- Download weights and cfg from [here](https://pjreddie.com/darknet/yolov2/) to model/ (only support yolov2-voc and yolov2-tiny-voc)


- Convert darknet weights and cfg to keras format
```bash
python convert_model yolov2-voc
```

- Detect Object in an image located at input/
```bash
python detection  yolov2-voc test1.jpg
```
- Detect Object in a video located at input/
```bash
python Main yolov2-voc 1.mp4
```
