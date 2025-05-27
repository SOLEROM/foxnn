# facenet full app from src

* (1) face detection model : retinaface-pytorch build retinaface.onnx
(conda retinaface)
* (2) Face Feature Extraction model : facenet
    * extract 128-dimensional features based on the input face image
    * calculate the Euclidean distance with the features 



## ref
* wiki instructions https://wiki.luckfox.com/Luckfox-Pico/Luckfox-Pico-RV1106/Luckfox-Pico-Ultra-W/Luckfox-Pico-RKNN-Test#4-model-deployment
* and [use the rknn2 toolkit](./rknn2.md)

 
## (1) retinaface

```
git clone https://github.com/bubbliiiing/retinaface-pytorch.git
cd retinaface-pytorch

conda create -n retinaface python=3.6

conda activate retinaface
pip install -r requirements.txt

```

```
//Execute the script to obtain the ONNX file
python export_onnx.py

```


output

```
model_data/
|-- Retinaface_mobilenet0.25.pth
|-- mobilenetV1X0.25_pretrain.tar
`-- retinaface.onnx

```

## (2) Feature Extraction

```
git clone https://github.com/bubbliiiing/facenet-pytorch.git
cd facenet-pytorch
conda create -n facenet python=3.6

conda activate facenet
pip install -r requirements.txt


```

add and run export py:

```
python export_onnx.py 
======================1
from nets.facenet import Facenet
from torch import onnx
import torch 

model_path='model_data/facenet_mobilenet.pth' # Model path 
model = Facenet(backbone="mobilenet",mode="predict",pretrained=True) # Model initialization
device = torch.device('cpu')
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
example=torch.rand(1,3,160,160) # Given input
torch.onnx.export(model,example,'model_data/facenet.onnx',verbose=True,opset_version=9) # Export
```

resluts:

```
model_data/
|-- backbone_weights_of_mobilenetv1.pth
|-- facenet.onnx
|-- facenet_mobilenet.pth
|-- lfw_pair.txt
`-- roc_test.png


```


## (3) convert

```
conda activate RKNN-Toolkit2

cd convert
python convert.py <onnx_model_path> <dataset_path> <output_model_path> <model_type>


    onnx_model_path: The path to the ONNX model exported during training. The example model is located in luckfox_onnx_to_rknn/model.
    dataset_path: Provide a small number of images as a reference for model conversion. The storage path of the images should be written in a txt file and passed as a parameter to the conversion script.
    output_model_path: The name and path of the exported RKNN model. Ensure it has a .rknn suffix.
    model_type: Provide different RKNN preprocessing settings based on the model type being converted. In this example, only the RetinaFace model requires special configuration during conversion.


```


run 

```
## RetinaFace Model Conversion
python convert.py ../model/retinaface.onnx ../dataset/retinaface_dataset.txt ../model/retinaface.rknn Retinaface


## Facenet Model Conversion
python convert.py ../model/facenet.onnx ../dataset/facenet_dataset.txt ../model/facenet.rknn Facenet
```



## test simulation on x86 docker (onnx)

```
conda activate RKNN-Toolkit2

cd /home/user/shared/faceIT/luckfox_pico_rknn_example/scripts/luckfox_onnx_to_rknn/sim


|-- facenet
|   |-- 001.jpg
|   |-- 002.jpg
|   |-- face.jpg
|   |-- facenet.py
|   |-- facenet_dataset.txt
|   |-- mobilefacenet.onnx
|   `-- requirements.txt
`-- retinaface
    |-- dataset.txt
    |-- result.jpg
    |-- retinaface.onnx
    |-- retinaface.py
    `-- test.jpg


cd sim/retinaface && python retinaface.py
cd sim/facenet && python facenet.py


```


## simulation embbedded part ????


https://wiki.luckfox.com/Luckfox-Pico/Luckfox-Pico-RV1106/Luckfox-Pico-Ultra-W/Luckfox-Pico-RKNN-Test/#542-software-simulator-model-verification  


TBD????