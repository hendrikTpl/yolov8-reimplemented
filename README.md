# yolov8-reimplemented

### Requirements
>> Python 3.8
```bash
python3.8 -m venv venv38 && source venv38/bin/activate
```
### Environments

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt 
```
### Pre-trained Weight
can be downloaded here
https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes


### Training

To train or fine-tune a model, use the `train.py` script. For example, to fine-tune the pretained YOLOv8n model on the coco128 dataset and save the weights:
```bash
python train.py --weights model/weights/yolov8n.pt \
                 --train-config model/config/training/fine_tune.yaml \
                 --dataset model/config/datasets/coco128.yaml \
                 --save
```

### Inference

To perform inference with a model, use the `inference.py` script. For example, to evaluate a model on a particular dataset:
```bash
python inference.py --config model/config/models/yolov8n.yaml \
                     --weights model/weights/yolov8n.pt \
                     --dataset model/config/datasets/coco8.yaml \
                     -v
```

#### ONNX
ONNX models are also supported for inference. To convert a YOLOv8 model to the ONNX format, use the `export_model.py` script as follows:
```bash
python export_model.py --config model/config/models/yolov8n.yaml \
                        --weights model/weights/yolov8n.pt \
                        --output model/weights/yolov8n.onnx
```

### References

