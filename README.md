# Image Classification Module

training and testing module for image classification task

## Data Preprocessing

以 dogs-vs-cats dataset 為例
整理 train.csv 跟 val.csv 兩份資料表單
格式如下

* img_path (string): 相對於照片的根目錄的路徑
  假設我們的資料集根目錄為 data/ 裡面有許多不同種類的資料集，則 img_path 裡面的值就要設 data/ 之後的相對路徑

```
data/ 
 | - cifar10/
 | - coco/
 | - dogs-vs-cats/
      | - train/
      | - test/   
```

* label (int): 類別定義

![](image/train_dataframe.png)
![](image/val_dataframe.png)
