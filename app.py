import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io

# Cargar el modelo YOLO
model_path = "C:/Users/User/Documents/Proyectos/Proyecto IA/yolov8_training_v7/weights/best.pt"
model = YOLO(model_path)

def draw_boxes_without_conf(img, result):
    """
    Dibuja en la imagen 'img' las cajas de detección obtenidas en 'result',
    mostrando únicamente el nombre de la clase sin la precisión, usando color rojo.
    """
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    names = model.names
    detected_classes = []

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        label = names[int(cls)]
        detected_classes.append(label)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    
    return img, detected_classes

# Interfaz de Streamlit
st.markdown("<h1 style='text-align: center;'>🌱 Detección de Enfermedades del Cacao 🍫</h1>", unsafe_allow_html=True)
st.header("📸 Sube una imagen")

imagen_subida = st.file_uploader("Elige una imagen de una hoja afectada o sana", type=["jpg", "png", "jpeg"])

if imagen_subida:
    # Leer la imagen directamente sin modificar su calidad
    img_bytes = imagen_subida.read()
    imagen = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Convertir a numpy sin modificar la calidad
    img_np = np.array(imagen)

    st.image(imagen, caption="📷 Imagen subida (calidad original)", use_container_width=True)

    st.info("🔍 Procesando la imagen con el modelo de IA...")

    # Guardar la imagen temporalmente en memoria sin comprimir
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(img_bytes)  # Guardar la imagen original sin alteraciones
    
    # Realizar la detección
    results = model(temp_path)
    
    # Convertir la imagen a formato BGR para OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Dibujar las cajas sin la precisión y obtener las enfermedades detectadas
    result_img, detected_classes = draw_boxes_without_conf(img_bgr, results[0])

    # Convertir la imagen resultante a RGB para mostrar en Streamlit sin pérdida de calidad
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    st.image(result_img_rgb, caption="📊 Resultado del análisis", use_container_width=True)

    # Mostrar enfermedades detectadas
    if detected_classes:
        enfermedades_detectadas = ", ".join(set(detected_classes))
        st.subheader("🦠 Enfermedades detectadas:")
        st.write(f"🔬 {enfermedades_detectadas}")
    else:
        st.subheader("✅ No se detectaron enfermedades en la imagen.")

    st.success("✅ Detección completada con éxito")

#streamlit run app.py