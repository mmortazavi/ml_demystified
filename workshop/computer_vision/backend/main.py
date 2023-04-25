
import os
import cv2
import base64
import numpy as np

import uvicorn
from fastapi import FastAPI, Request
from fastapi import HTTPException, status


from helper import setup_logger
from helper import load_yolov8_model, object_detector_yolov8
from datamodels import ImageItem, SPADResponse


# --------------------------------- Variables -----------------------------
device = "0" # for cpu: "cpu" # for gpu: "0"
task = "segment"
model_name = "spad"
model_version = "v1.0.0"
model_filename = "best.pt"
path_to_model = r"C:\Users\majmo\Git\ml_demystified\workshop\computer_vision\models"
path_to_data = r"C:\Users\majmo\projects\ikiu\backend"

model_threshold = 0.5
api_key = "4DGn!CK6ET6%7xB&eiqdtk"

# --------------------------------- Loggers --------------------------------
path_to_logs = os.path.join(path_to_data, "logs")
path_to_images = os.path.join(path_to_data, "images")

if not (os.path.exists(os.path.join(path_to_logs))):
    os.mkdir(os.path.join(path_to_logs))
if not (os.path.exists(path_to_images)):
    os.mkdir(path_to_images)

path_to_logger = os.path.join(path_to_logs, "SPD-App.log")
logger = setup_logger("SPD-App", path_to_logger)

# --------------------------------- Models --------------------------------
model = load_yolov8_model(path_to_model, model_name, model_version, task)


# ----------------------------------------------------------------------------#
# --------------------------------- Flask Apps --------------------------------
app = FastAPI()

@app.post('/spad/predict', response_model = SPADResponse)
def object_detection(item: ImageItem, request: Request):
    
    """
    Endpoint for performing object detection on an image and returning the detection results.
    
    Args:
        item (ImageItem): A Pydantic model that represents the input image as a base64-encoded string.
        request (Request): A FastAPI Request object representing the incoming HTTP request.
    
    Returns:
        A Pydantic model `SPADResponse` that includes a base64-encoded string of the output image and a list of 
        detection results represented as dictionaries in JSON format.
    """
    
    logger.info(f"Performing Magic..")

    if request.headers['API-KEY'] == api_key:
        
        img_byte = base64.b64decode(item.Imageb64)
        nparr = np.frombuffer(img_byte, np.byte)
        image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

        df_response, img_response = object_detector_yolov8(model, image, model_name, model_threshold, device)

        # Convert the numpy array to a byte string
        img_response_bytes = img_response.tobytes()

        # Encode the byte string as a base64 string
        img_response_str = base64.b64encode(img_response_bytes)#.decode('utf-8')

        logger.info("The Magic Completed!")

        return SPADResponse(
                            Imageb64= img_response_str,
                            Response = df_response.to_dict('records')

                            )

    else:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect Api-key",
        )
# ----------------------------------------------------------------------------#


if __name__ == "__main__":

    try:
        uvicorn.run(app, host='0.0.0.0', port=2023, debug=True)
    except Exception as e:
        logger.info(f"Exception at App: {e}")