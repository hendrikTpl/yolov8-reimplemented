# yolov8-reimplemented

### Requirements
>> Python 3.8

### Environments

```bash
pip install -r requirements.txt
```

### Training

To train or fine-tune a model, use the `train.py` script. For example, to fine-tune the pretained YOLOv8n model on the coco128 dataset and save the weights:
```bash
python3 train.py --weights model/weights/yolov8n.pt \
                 --train-config model/config/training/fine_tune.yaml \
                 --dataset model/config/datasets/coco128.yaml \
                 --save
```

### Inference

To perform inference with a model, use the `inference.py` script. For example, to evaluate a model on a particular dataset:
```bash
python3 inference.py --config model/config/models/yolov8n.yaml \
                     --weights model/weights/yolov8n.pt \
                     --dataset model/config/datasets/coco8.yaml \
                     -v
```

#### ONNX
ONNX models are also supported for inference. To convert a YOLOv8 model to the ONNX format, use the `export_model.py` script as follows:
```bash
python3 export_model.py --config model/config/models/yolov8n.yaml \
                        --weights model/weights/yolov8n.pt \
                        --output model/weights/yolov8n.onnx
```

### References

