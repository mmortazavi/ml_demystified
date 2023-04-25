import os
from pathlib import Path
from yolov5.segment import train
from yolov5.utils import general



# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

PROJECT_ROOT = Path( __file__ ).parent.absolute() #os.path.realpath(__file__) #Path.cwd()
train.ROOT = PROJECT_ROOT
general.ROOT = PROJECT_ROOT


data = r"C:\Users\majmo\Git\ml_demystified\workshop\computer_vision\custom\data\data.yaml"
weights = r'..\models\yolov5m-seg.pt'
project = r"C:\Users\majmo\Git\ml_demystified\workshop\computer_vision\custom"


if __name__ == '__main__':

    train.run(imgsz=320,
            batch_size=16,
            epochs = 200,
            save_period = -1,
            weights=weights,
            data=data,
            project=project,  # save results to project/name
            device = "0", 
            name='seg_run')