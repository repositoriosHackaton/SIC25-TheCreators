#Testeo del modelo

from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow 

model = YOLO('/content/runs/detect/train4/weights/best.pt')  

# Cargar la imagen 
image_path = '/content/drive/My Drive/moniliasis-on-cocoa-global-2.jpg'  
image = cv2.imread(image_path)

# Realiza la detección
results = model(image)

# Visualizacion de los resultados
for result in results:
    boxes = result.boxes.cpu().numpy()  
    for box in boxes:
        r = box.xyxy[0].astype(int)  
        cv2.rectangle(image, r[:2], r[2:], (0, 255, 0), 2)  
        class_id = int(box.cls[0])  
        class_name = model.names[class_id]  
        confidence = box.conf[0]  
        label = f'{class_name}: {confidence:.2f}' 
        cv2.putText(image, label, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
# Muestra la imagen con las detecciones
cv2_imshow(image) 

# Guardar la imagen con las detecciones
cv2.imwrite('/content/detecciones.jpg', image)