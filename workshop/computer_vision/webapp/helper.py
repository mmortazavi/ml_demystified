import os
import logging
import numpy as np
import pandas as pd

from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler

from yolov5 import YOLOv5
from ultralytics import YOLO


def setup_logger(logger_name, log_file, time_base=False, level=logging.INFO):

    def namer(name):
        return name.replace(".csv", "") + ".csv"

    """[summary]

    Args:
        logger_name ([type]): [description]
        log_file ([type]): [description]
        time_base (bool, optional): [description]. Defaults to False.
        level ([type], optional): [description]. Defaults to logging.INFO.

    Returns:
        [type]: [description]
    """

    if time_base:
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter("%(asctime)s;%(message)s")
        
        fileHandler = TimedRotatingFileHandler(
            log_file, when="midnight", interval=1
        )  # "midnight"
        fileHandler.suffix = "%Y%m%d"
        # fileHandler.suffix = "%Y-%m-%d_%H-%M-%S"
        fileHandler.namer = (namer)  # <-- Here's where I assign the custom namer.
        fileHandler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(fileHandler)

        return logger
    else:

        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter("%(asctime)s;%(message)s")
        
        fileHandler = RotatingFileHandler(
            log_file, mode="w", maxBytes=5000000, backupCount=1
        )
        fileHandler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(fileHandler)

        return logger
  
def load_yolov5_model(path_to_model: str, model_name:str, model_version:str, device: str):

    try:
        # Load Yolo Custom Model
        print(f"Loading Yolo {model_name}...")
        full_path_to_model = os.path.join(path_to_model, model_name, model_version, 'best.pt')
        # init yolov5 model
        model = YOLOv5(full_path_to_model, device)
        # model = torch.hub.load(path_to_yolo, 'custom', path=full_path_to_model, source="local", force_reload=True, _verbose=False)     
        print(f"Yolo {model_name} loaded!")
        return model
    except Exception as e:
        print(f"Exception at Yolo {e}")

def load_yolov8_model(path_to_model: str, model_name:str, model_version:str, task: str):

    try:
        # Load Yolo Custom Model
        print(f"Loading Yolo {model_name}...")
        full_path_to_model = os.path.join(path_to_model, model_name, model_version, 'best.pt')
        # init yolov5 model
        model = YOLO(full_path_to_model, task)

        # model = torch.hub.load(path_to_yolo, 'custom', path=full_path_to_model, source="local", force_reload=True, _verbose=False)     
        print(f"Yolo {model_name} loaded!")
        return model
    except Exception as e:
        print(f"Exception at Yolo {e}")
        
def object_detector_yolov8(model, image: np.array, model_name:str, confidence_value: int, device: str):
    
    results = model.predict(source= image, conf = confidence_value, device= device)
    results = results[0]
    
    try:
        response = []

        names = results.names
        for result in results:
            boxes_unravelled = [value for value in result.boxes.xyxy[0].detach().cpu().numpy()] # Boxes object for bbox outputs
            cls_id = int(result.boxes.cls[0])
            cls_name = names[int(result.boxes.cls[0])]
            conf   = result.boxes.conf[0].item()    # Boxes object for bbox outputs
            
            boxes_unravelled = boxes_unravelled + [conf, cls_id, cls_name]

            response.append(boxes_unravelled)

        response_df = pd.DataFrame(response, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])
        return response_df, results.plot()

    except Exception as e:
        print(f"Exception at Yolo Prediction {e}")
           
def object_detector_yolov5(model, image: np.array, model_name:str, confidence_value: int):
    try:
        model.model.conf = confidence_value
        print(f"Predicting using {model_name}")
        response = model.predict(image)  # includes NMS
        if response.pred[0].size()[0] !=0:
            print(f"Predicted Completed by {model_name}!")
            return response
        else:
            return None

    except Exception as e:
        print(f"Exception at Yolo Prediction {e}")

def process_yolo_response(response: pd.DataFrame, model_name:str) -> pd.DataFrame:

    if response:
        print(f"Process Yolo Response for {model_name}")
        return response.pandas().xyxy[0]
    else:
        return pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'])



