# INSTALLING 

### 1. REQUIRED DEPENDENCIES
- python 3 
- tensorflow cpu or gpu >= 1.5.0

### 2. INSTALL DEPENDENCIES USING PIP3:
````
pip3 install tensorflow-gpu
pip3 install Cython
pip3 install pillow
pip3 install lxml
pip3 install jupyter
pip3 install matplotlib
````

### 3. INSTALL PROTOC
- DOWNLOAD : https://github.com/google/protobuf/releases/tag/v3.4.0
- SET ENVIRONMENT VARIABLE

# SETTING UP

### 1. SET PATH
- REPLACE "YOUR_SYSTEM_PATH"
````
set PYTHONPATH=YOUR_SYSTEM_PATH/models;YOUR_SYSTEM_PATH/models/research;YOUR_SYSTEM_PATH/models/research/slim

set PATH=%PATH%%PYTHONPATH%
````

### 2. PROTOC - GO TO models/research
````
protoc ./object_detection/protos/*.proto --python_out=.
````

### 3. BUILD & INSTALL
````
py -3 setup.py build
py -3 setup.py install
````

### 4. TEST IF IT WORKING OR NOT - GO TO models/research/object_detection
````
jupyter notebook object_detection_tutorial.ipynb
````

# PREPARING TO TRAIN

### 1. COLLECT IMAGES
- LABEL THE IMAGES USING : https://github.com/tzutalin/labelImg
- PUT IMAGES AT models/research/object_detection/images
- DIVIDE THE IMAGES INTO train AND test
````
./images
--/test
--/train
````

### 2. GENERATE CSV - GO TO models/research/object_detection
````
py -3 xml_to_csv.py
````

### 7. GENERATE TFRECORD
- OPEN generate_tfrecord.py AND EDIT THE LABEL MAP 
````
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'one':
        return 1
    elif row_label == 'two':
        return 2
    elif row_label == 'three':
        return 3
    else:
        None
````
THEN,
````
py -3 generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

py -3 generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
````

### 8. CREATE label_map.pbtxt
````
item {
  id: 1
  name: 'one'
}

item {
  id: 2
  name: 'two'
}

item {
  id: 3
  name: 'three'
}
````
- EXAMPLE AT models/research/object_detection/data
- PUT label_map.pbtxt IN THE models/research/object_detection/training

### 9. CONFIGURE TRAINING

#### 9.1 CONFIGURE MODEL
- COPY THE CONFIG FILE models/research/object_detection/samples/configs/XXXX.config
- PUT THE CONFIG IN THE models/research/object_detection/training

#### 9.2 FINE TURN MODEL
- DOWNLOAD : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
- PUT THE FINE TUNE MODEL IN THE models/research/object_detection

### 10. EDIT CONFIG FILE
- num_classes = NUM OF CLASSES
- REPLACE all "PATH_TO_BE_CONFIGURED"
- eval_config>num_examples = NUMBER OF TEST IMAGES
- max_detections_per_class = 1
- max_total_detections = 1

### 11. TRAINING 
- REPLACE XXXX
````
py -3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/XXXX.config
````

### 12. EVALUATION
- REPLACE XXXX
````
py -3 eval.py --logtostderr --train_dir=training/ --pipeline_config_path=training/XXXX.config --checkpoint_dir=training/ --eval_dir=training/
````

### TENSORBOARD
````
tensorboard --logdir=training
````

### 13. EXPORT INFERENCE GRAPH 
- REPLACE XXXX and YYYY
````
py -3 export_inference_graph.py --input_type image_tensor --pipeline_config_path training/XXXX.config --trained_checkpoint_prefix training/model.ckpt-YYYY --output_directory inference_graph
````

## ERRORS
- "ValueError: Tried to convert 't' to a tensor and failed. Error: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted []" = ref : https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/issues/11

## REF
- https://github.com/tensorflow/models/tree/master/research/object_detection
- https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
