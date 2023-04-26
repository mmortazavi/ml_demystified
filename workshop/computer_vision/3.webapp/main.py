import os
import cv2
from PIL import Image

from datetime import datetime
import matplotlib.pyplot as plt

import mpld3
import streamlit as st
import streamlit.components.v1 as components

import helper


model_namespace = {"spad": "Solar Panel Array Detector"}
model_description = {"spad": "Detect Solar Panel Arrays in Aerial Images"}

st.set_page_config(page_title="SPAD App", layout="wide")

# --------------------------------- Variables --------------------------------
device = "0" #"cpu" # "cuda:0" # or "cpu"        
task = "segment"
path_to_data = r"C:\Users\majmo\projects\ikiu"
path_to_models = r"..\models"

# --------------------------------- Paths --------------------------------
list_of_models = os.listdir(path_to_models)
list_of_models_namespace = [x if x not in model_namespace else model_namespace[x] for x in list_of_models]

if not (os.path.isdir(os.path.join(path_to_data, "outputs"))):
    os.mkdir(os.path.join(path_to_data, "outputs"))

if not (os.path.isdir(os.path.join(path_to_data, "uploads"))):
    os.mkdir(os.path.join(path_to_data, "uploads")) 
    
# --------------------------------- Models --------------------------------

@st.cache_resource
def load_model(model_name, model_version, task):
    try:
        return helper.load_yolov8_model(path_to_models, model_name, model_version, task)
    except Exception as e:
        print(f"Exception at loading {model_name} model: {e}")
        raise e

# --------------------------------- Steamlit Webapp --------------------------------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()};font-size:28px</style>', unsafe_allow_html=True)
         
local_css("style.css")
            
col1, col2 = st.columns([1, 2.5])
widget_id = (id for id in range(1, 100_00))

with col1:
    
    st.title("A Visual Model Evaluator")
    st.write("A simple webapp for visually evaluating object detection models")
    
    st.markdown(f'<h1 style="color:#6495ED;font-size:24px;">{"1. Model Selection:"}</h1>', unsafe_allow_html=True)

    model_selection_namespace = st.selectbox("Model:", list_of_models_namespace) 
    model_selection = [k for k in model_namespace.keys() if model_namespace[k] == model_selection_namespace][0]
    
    st.write(f'{model_description[model_selection]}')

    list_model_versions = os.listdir(os.path.join(path_to_models, model_selection))
    model_version = st.selectbox("Version:", list_model_versions) 
    
    st.markdown(f'<h1 style="color:#6495ED;font-size:24px;">{"2. Choose Image:"}</h1>', unsafe_allow_html=True)
    image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'], accept_multiple_files=False)
    ts = datetime.timestamp(datetime.now())
    
    st.markdown(f'<h1 style="color:#6495ED;font-size:24px;">{"3. Choose Confidence:"}</h1>', unsafe_allow_html=True)
    conf_slider = st.slider(label='Confidence threshold (minimum acceptable confidence level for displaying a bounding box):', value = 0.5, min_value = 0.0, max_value =1.0,  step = 0.1)

    # Start button
    st.markdown(f'<h1 style="color:#6495ED;font-size:24px;">{"4. Start Inspection:"}</h1>', unsafe_allow_html=True)

    if st.button('Start'):
        
        st.markdown(f'<h1 style="color:#6495ED;font-size:20px;text-align: left">{"Status:"}</h1>', unsafe_allow_html=True)
        status_bar = st.progress(0, text = "Start Inspecting...")
        
        # Load Model

        model = load_model(model_selection, model_version, device)
        status_bar.progress(20, text = f"Loading Model: {model_selection_namespace}")

        with col2:

            if image_file is not None:
                
                # Construct Inout and Output Image Name
                imgpath = os.path.join(path_to_data, "uploads", str(ts) + "_" + image_file.name )
                outputpath = os.path.join(path_to_data, "outputs", os.path.basename(imgpath))
                
                with open(imgpath, mode="wb") as f:
                    f.write(image_file.getbuffer())
                
                # Fixing Channel Swap
                image_numpy = cv2.imread(imgpath)#[..., ::-1]
                
                try:
                    # YoloV5 Predection
                    status_bar.progress(50, text = f"Predicting using YoloV8...")
                    df_response, img_response = helper.object_detector_yolov8(model, image_numpy, model_selection, conf_slider, device)
                    if not df_response.empty:
                        # Display Predection
                        status_bar.progress(70, text = f"Preparing to Display Results...")
                        subcol1, subcol2 = col2.columns([2,1])
                        with subcol1:

                            img_ = img_response[..., ::-1]
                            st.markdown(f'<h1 style="color:#6495ED;font-size:28px;">{f"5. {model_selection_namespace} Prediction:"}</h1>', unsafe_allow_html=True)
                            st.image(img_, caption='Model Prediction(s)', width=800)
                            
                            # YoloV5 Dataframe Predection
                            st.markdown(f'<h1 style="color:#6495ED;font-size:28px;">{"6. Resposne Dataframe"}</h1>', unsafe_allow_html=True)
                            df_response = df_response[['name', "confidence", 'xmin', 'ymin', 'xmax', 'ymax']]
                            st.dataframe(df_response.style.set_properties(**{"background-color": "white", "color": "black"}), use_container_width=True)
                            status_bar.progress(90, text = f"Constructing Resposne Dataframe...")
                            
                        with subcol2:

                            ## Histogram of Confidence Levels
                            # st.write('### Histogram of Confidence Levels')
                            st.markdown(f'<h1 style="color:#6495ED;font-size:28px;">{"7. Confidence Levels"}</h1>', unsafe_allow_html=True)

                            fig = plt.figure() 
                            ax = df_response["confidence"].plot.hist(bins=10, alpha=0.5, figsize=(5, 2))
                            fig_html = mpld3.fig_to_html(fig)
                            components.html(fig_html, height=400)
                            status_bar.progress(100, text = f"Drawing Histogram of Confidence Levels...")
                    else:
                        status_bar.progress(100, text = f"Nothing Predecited!")

                except Exception as e:
                    print(f'Exception at Main Loop: {e}')
                
        