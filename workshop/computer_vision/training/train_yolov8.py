from ultralytics import YOLO


data = r"C:\Users\majmo\Git\ml_demystified\workshop\computer_vision\custom\data\data.yaml"
weights = r'C:\Users\majmo\Git\ml_demystified\workshop\computer_vision\playground\models\yolov8m-seg.pt'
project = r"C:\Users\majmo\Git\ml_demystified\workshop\computer_vision\custom"

# Load a model
model = YOLO(weights, "segment")  # load a pretrained model (recommended for training)

YOLO
if __name__ == '__main__':

    # Use the model
    model.train(imgsz=320,
                batch=8,
                epochs = 5,
                save_period = -1,
                model=weights,
                data=data,
                project=project,  # save results to project/name
                device = "0", 
                name='seg_yolov8_run',
                callback= "mlflow")