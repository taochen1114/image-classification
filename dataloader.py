import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transform
from PIL import Image
import pandas as pd
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

def build_augmentor():
    ia.seed(1)
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [            
            sometimes(iaa.Affine(
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-10, 10),
                shear=(-5, 5),
                order=1,
                name='Affine'
            )),

            # sharpen
            sometimes(iaa.Sharpen(alpha=(0, 0.2))),

            # contrast
            sometimes(iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)),
            
            # brightness
            sometimes(iaa.Multiply((0.9, 1.1), name='Brightness')),
            
            # blur
            sometimes(iaa.GaussianBlur((0, 3.0))),
        ],
        random_order=True
    )
    return seq

def run_augmentation(img, seq):
    # run augmentation
    img_aug = seq.augment_images(img)      
    return img_aug
    
class ClassificationImage(Dataset):
    
    def __init__(self, data_dir, num_classes, csv_path, loss, aug, transform=None):

        self.data_dir = data_dir
        self.num_classes = num_classes
        
        tmp_df = pd.read_csv(csv_path)        
        self.images = tmp_df['img_path']
        self.labels = tmp_df['label']
        
        self.loss = loss
        self.transform = transform
        
        self.aug = aug
        
        if self.aug:
            self.seq = build_augmentor()
            
        print('Total number of data: {}'.format(self.images.shape[0]))

    def __getitem__(self, index):

        img = Image.open(os.path.join(self.data_dir, self.images[index]))
        img = img.convert('RGB')
        
        # Data augmentation
        if self.aug:
            arr = np.expand_dims(np.array(img), axis=0)
            arr = run_augmentation(img=arr, seq=self.seq).squeeze(axis=0)
            img = Image.fromarray(np.uint8(arr))

        if self.transform is not None:
            img = self.transform(img)

        label_tensor = torch.LongTensor(np.array([self.labels[index]]))

        label_onehot = torch.FloatTensor(self.num_classes)
        label_onehot.zero_()
        label_onehot.scatter_(0, label_tensor, 1)

        return self.images[index], img, label_onehot.squeeze()

    def __len__(self):
        return self.images.shape[0]
   
   
def get_loader(args,
               image_mean=(0.485, 0.456, 0.406),
               image_std=(0.229, 0.224, 0.225),
               image_scale=(224, 224)):

    # Image transformations
    if args.arch == 'inception_v3':
        image_scale = (299, 299)
        
    resize_wh = image_scale[0]+int(image_scale[0]*0.1)
    image_resize_scale = (resize_wh, resize_wh)
    
    train_transform = transform.Compose([transform.Resize(image_resize_scale),
                                    transform.RandomCrop(image_scale),
                                    transform.RandomHorizontalFlip(0.5),
                                    transform.ToTensor(),
                                    transform.Normalize(image_mean, image_std),
                                    ])
    test_transform = transform.Compose([transform.Resize(image_scale),
                                   transform.ToTensor(),
                                   transform.Normalize(image_mean, image_std),
                                   ])

    if not args.test_mode: 
        print("Training Mode, loading data ...")
        
        dataset_train = ClassificationImage(args.data_dir, args.num_classes, args.train, args.loss, args.aug, train_transform)
        train_loader = DataLoader(dataset_train,
                                    batch_size=args.batch,
                                    shuffle=True,
                                    num_workers=args.workers,
                                    )
        print('Trainining loader is ready.')

        # Validation dataloader
        dataset_val = ClassificationImage(args.data_dir, args.num_classes, args.val, args.loss, False, test_transform)
        val_loader = DataLoader(dataset_val,
                                batch_size=args.batch,
                                shuffle=False,
                                num_workers=args.workers,
                                )
        print('Validation loader is ready.')
        
        return (train_loader, val_loader)
    else:
        print("Testing Mode, loading data ...")
        dataset_test = ClassificationImage(args.data_dir, args.num_classes, args.test, args.loss, False, test_transform)
        test_loader = DataLoader(dataset_test,
                                 batch_size=args.batch,
                                 shuffle=False,
                                 num_workers=args.workers,
                                 )

        print('Test loader is ready.')
        return test_loader

    


