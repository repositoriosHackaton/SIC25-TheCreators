#Entrenamiento en Google Colab

from google.colab import drive
drive.mount('/content/drive')

# Instalar las librerías roboflow y ultralytics
!pip install roboflow ultralytics

# Importar las librerías
from roboflow import Roboflow
from ultralytics import YOLO

# Configurar Roboflow
rf = Roboflow(api_key="AqQkYHvC0QlIDFYZ8We2")
project = rf.workspace("cacao-mev33").project("cacao-dataset-wb2hr")
dataset = project.version(4).download("yolov9")

# Cargado del modelo YOLOv9t
model = YOLO('yolov9m.pt')

# Entrenamiento del modelo
model.train(data=dataset.location + "/data.yaml", epochs=60, imgsz=600, freeze = 10, patience = 10, batch=12)

# Evaluar el modelo
metrics = model.val()


