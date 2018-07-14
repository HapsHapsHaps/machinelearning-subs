echo Fixing COCO API and dependencies
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
make install
cp -r pycocotools /tensorflow/tensorflow/models/research/

cd /tensorflow/tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.

