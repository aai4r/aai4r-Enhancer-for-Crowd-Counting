# aai4r-Enhancer-for-Crowd-Counting

## Installation
* Clone this repo into a directory
* Install Python dependencies. We use python 3.8.5 and pytorch 1.7.1
<!-- ```
pip install -r requirements.txt
``` -->
* Download ShanghaiTech dataset from [GoogleDrive](https://drive.google.com/drive/folders/17WobgYjekLTq3QIRW3wPyNByq9NJTmZ9?usp=sharing)
* Download models from [GoogleDrive](https://drive.google.com/drive/folders/1sXa9GcwA2d5mLhMRDK37yQAvdsZKcwh4?usp=sharing)

## Preparation
Organizing the datas and models as following:
```
EPFSNet/
        |->datas/
        |    |->part_B_final/
        |    |    |->train_data/
        |    |    |    |->images/
        |    |    |    |->ground_truth/
        |    |    |->val_data/
        |    |    |    |->images/
        |    |    |    |->ground_truth/
        |    |    |->test_data/
        |    |    |    |->images/
        |    |    |    |->ground_truth/
        |    |->...
        |->models/
        |    |->PFSNet_PartB.pth
        |    |->EPFSNet_PartB.pth
        |    |->...
        |->datasets/
        |    |->dataloader.py
        |->save/
        |->main.py
        |->prepare_dataset.py
        |->model.py
```
Generating the density maps for the data:
```
python prepare_dataset.py --data_path './datas/part_B_final'
```


## Training
Run the following commands to launch training PFSNet:
```
python main.py --data_path './datas/part_B_final' --save_path './save' --learning_rate 5e-5 --level_counting_loss_ratio 0.5 
```
Run the following commands to launch training Image Enhancer:
```
python main.py --data_path './datas/part_B_final' --save_path './save' --enhancer 'yes' --model_path './models/PFSNet_PartB.pth' --learning_rate 5e-5 --repeat 10
```


## Testing
Run the following commands to launch inference:
```
python main.py --data_path './datas/part_B_final' --model_path './models/EPFSNet_PartB.pth' --enhancer 'yes' --step 'test'
```