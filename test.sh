## test model_v01
# python inference.py  \
#     --data-dir data/  \
#     --test-mode \
#     --test data/dogs-vs-cats/test.csv  \
#     --arch resnet18  \
#     -c model_ckpt/model_v01/  \
#     --model-file model_best.pth.tar \
#     --num-classes 2  \
#     --class-def data/dogs-vs-cats/class_def.txt  

## test model_v02
python inference.py  \
    --data-dir data/  \
    --test-mode \
    --test data/dogs-vs-cats/test.csv  \
    --arch resnet18  \
    -c model_ckpt/model_v02/  \
    --model-file model_best.pth.tar \
    --num-classes 2  \
    --class-def data/dogs-vs-cats/class_def.txt  
