## model_v01 resnet18 
# python main.py \
#         --data-dir data/  \
#         --train data/dogs-vs-cats/train.csv  \
#         --val data/dogs-vs-cats/val.csv  \
#         --arch resnet18 \
#         --num-classes 2 \
#         --epochs 50 \ 
#         --lr 0.001 \
#         --batch 32 \
#         --checkpoint model_ckpt/model_v01/  \
#         --aug True


## model_v02 resnet18 with pretrain weights and lr decay schedule
python main.py \
        --data-dir data/  \
        --train data/dogs-vs-cats/train.csv  \
        --val data/dogs-vs-cats/val.csv  \
        --arch resnet18 \
        --num-classes 2 \
        --epochs 30 \
        --lr 0.001 \
        --batch 32 \
        --checkpoint model_ckpt/model_v02/  \
        --gamma 0.8 --schedule 10 20 \
        --aug --pretrain 

