#Guardado de la carpeta de entrenamiento

import shutil
import os

ruta_entrenamiento = 'runs/detect/train4'
ruta_destino = '/content/drive/My Drive/YOLOv9_v2'
os.makedirs(ruta_destino, exist_ok=True)
shutil.copytree(ruta_entrenamiento, os.path.join(ruta_destino, os.path.basename(ruta_entrenamiento)))

print(f"Resultados del entrenamiento copiados a: {os.path.join(ruta_destino, os.path.basename(ruta_entrenamiento))}")

