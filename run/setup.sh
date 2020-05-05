pip install -r requirements.txt
mkdir data
mkdir save
mkdir -p pretrained/imagenet
cd  pretrained/imagenet
wget https://hanlab.mit.edu/files/haq/mobilenetv2-150.pth.tar
cd ../..

