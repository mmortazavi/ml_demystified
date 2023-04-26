import os
import logging
import numpy as np
import pandas as pd

from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler

from yolov5 import YOLOv5
from ultralytics import YOLO


def setup_logger(logger_name, log_file, time_base=False, level=logging.INFO):

    """
    Creates and returns a logger object for writing log messages to a file.

    Args:
        logger_name (str): The name of the logger object.
        log_file (str): The path to the log file.
        time_base (bool, optional): If True, the log file will be rotated on a daily basis, with a
            suffix indicating the date of the current day. If False, the log file will be overwritten
            each time the program is run. Defaults to False.
        level (int, optional): The logging level to use for the logger object. Defaults to logging.INFO.

    Returns:
        logger: A logger object that can be used to write log messages to the specified file.

    Raises:
        ValueError: If the log file path is not valid.

    """
    
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

    """
    Loads a custom YOLOv5 model from a specified file path, and returns the model object.
    
    Args:
    - path_to_model (str): The file path to the directory containing the model file.
    - model_name (str): The name of the YOLOv5 model to load.
    - model_version (str): The version of the YOLOv5 model to load.
    - device (str): The device to load the model onto (e.g., 'cpu' or 'cuda').

    Returns:
    - model (YOLOv5): The loaded YOLOv5 model object.

    Raises:
    - Exception: If an error occurs while loading the model.
    """
    
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
    
    """
    Load a YOLOv8 model from disk and return it.

    Args:
        path_to_model (str): The path to the directory containing the saved model.
        model_name (str): The name of the model to load.
        model_version (str): The version of the model to load.
        task (str): The task that the model was trained on.

    Returns:
        A YOLOv8 model loaded from the specified path.

    Raises:
        Exception: If the model fails to load for any reason.

    Example usage:
        >>> model = load_yolov8_model('/path/to/models', 'yolov8', 'v1', 'object_detection')
    """
    
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
    
    """
    Uses a YOLOv8 model to detect objects in an input image and returns the detection results.

    Args:
        model (YOLO): A YOLOv8 model instance for object detection.
        image (np.array): An input image in the form of a NumPy array.
        model_name (str): The name of the YOLOv8 model used for detection.
        confidence_value (int): A threshold value for object detection confidence.
        device (str): The device used for inference, either "cpu" or "cuda".

    Returns:
        response_df (pd.DataFrame): A pandas DataFrame containing the detection results
            with columns for the bounding box coordinates, detection confidence,
            class ID, and class name.
        plot (Plot): An output image with the detection results plotted on top of
            the input image.
    """
    
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

        response_df = pd.DataFrame(response, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class_id', 'name'])
        return response_df, results.plot()

    except Exception as e:
        print(f"Exception at Yolo Prediction {e}")
           
def object_detector_yolov5(model, image: np.array, model_name:str, confidence_value: int):
    
    """Detect objects in an image using a YOLOv5 model.

    Args:
        model: A trained YOLOv5 model.
        image (numpy array): An image to detect objects in.
        model_name (str): Name of the YOLOv5 model being used.
        confidence_value (int): Confidence threshold for object detection.

    Returns:
        If objects are detected in the image, returns the response from the YOLOv5 model,
        otherwise returns None.
    """
    
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



