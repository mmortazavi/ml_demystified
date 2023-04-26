## Workshop Preparation
### 1. Tools
    1.1 [Visual Studio Code](https://code.visualstudio.com/) -> Install Python Extension
    1.2 [Python 3.9.13 ](https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe)
    1.3 Annotation Tools (alternative options, install one)
        1.3.1 [Label Studio ](https://labelstud.io/) 
        1.3.1 [LabelImg](https://github.com/heartexlabs/labelImg) 
        1.3.1 [OpenLabeler ](https://github.com/kinhong/OpenLabeler) 


### 2. Verify Python Installation:
    2.1 open Windows Terminal (or Windows Power Shell)
    2.2 Type, "python", and you should successfully land into Python 3.9.13

### 3. Prepare Working Directory (Python Env.):
    3.1 create a folder e.g. named segmentation
    3.2 open Windows Termianl (or Windows Power Shell)
    3.3 got to segmentation folder
    3.4 create a python virtual environment -> python -m venv venv
    3.5 activate the venv environment - > .\venv\Scripts\Activate.ps1
    3.6 Install yolov5 -> pip install yolov5 (https://pypi.org/project/yolov5/)
    3.7 Install FastAPI (https://fastapi.tiangolo.com/) -> pip install fastapi
    3.8 Install Streamlit (https://streamlit.io/) -> pip install streamlit

### 4. Project Structure:
    .
    ├── computer_vision
    │   │
    │   ├── 1. playground
    │   │   
    │   ├── 2. training
    │   │   
    │   ├── 3. webapp
    │   │  
    │   └── 4. backend
    │
    └── README.md

    each module contains:

    1. `playground`: This module is  where you will start experimenting with different models and techniques for segmentation or object detection. It includes an introductory Jupyter notebook for running inference.

    2. `training`: This module is where you will train your segmentation or object detection model. It may include scripts for data preparation, model training, and evaluation.

    3. `webapp`: This module is where you will develop the front-end of your computer vision application in Streamlit with some exemplary simple webapps. The main webapp include a web interface for uploading images or videos, viewing the results of the segmentation or object detection model, and interacting with the model in real time.

    4. `backend`: This module is where you will develop the back-end api of your computer vision application in FastAPI. A hello-work example is provided. This may include scripts for deploying the model, integrating it with the web application, and handling requests and responses.

### 5. [Workshop Slack Channel]( https://app.slack.com/client/T054GBG4ZJP/C054GF4TU02)

