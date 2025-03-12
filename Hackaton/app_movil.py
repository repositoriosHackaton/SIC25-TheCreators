import flet as ft
import torch
from PIL import Image, ImageDraw
import cv2
from ultralytics import YOLO
import os

# Modelo preentrenado
model_path = "C:/Users/User/Documents/Proyectos/Proyecto IA/Cacao_app/apk2/assets/best.pt"
model = YOLO(model_path)

def predict_disease(image_path):
    results = model(image_path)
    return results  

def show_results_page(page, image_path, results):
    page.clean()  

    # Cargar la imagen usando PIL
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return  

    # Dibujar las cajas delimitadoras
    if results:
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    print(f"Caja delimitadora: {xyxy}") # Imprimir las coordenadas de la caja
                    draw.rectangle(xyxy, outline="blue", width=10) 
            else:
                print("No se encontraron cajas delimitadoras en los resultados.")
    else:
        print("No se recibieron resultados del modelo.")

    # Guardar la imagen modificada
    modified_image_path = "modified_image.jpg"
    try:
        img.save(modified_image_path)
        print(f"Imagen guardada en: {modified_image_path}") 
    except Exception as e:
        print(f"Error al guardar la imagen: {e}")
        return  

    
    image_container = ft.Container(
        content=ft.Image(src=modified_image_path, fit=ft.ImageFit.CONTAIN),
        width=300,
        height=200,
        bgcolor="#FFF5EB",
        border_radius=10,
        shadow=ft.BoxShadow(blur_radius=5, color=ft.colors.BLACK26),
        alignment=ft.alignment.center,
        margin=ft.margin.symmetric(vertical=20)
    )

    result_text_container = ft.Container(
        content=ft.Text(
            f"Resultado: {', '.join(model.names[int(box.cls[0].item())] for result in results if result.boxes is not None for box in result.boxes)}",
            size=30,
            weight=ft.FontWeight.BOLD,
            color="#3D1E10",
            text_align=ft.TextAlign.CENTER,
        ),
        margin=ft.margin.symmetric(vertical=20)
    )

    back_button = ft.ElevatedButton(
        text="VOLVER",
        icon=ft.icons.ARROW_BACK,
        on_click=lambda _: show_main_page(page),
        bgcolor="#3D1E10",
        color="#FFE4C4",
        width=200,
        height=50,
        icon_color="#FFE4C4",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10)
        )
    )

    page.add(
        ft.SafeArea(
            ft.Column(
                controls=[
                    ft.Container( # Container que envuelve el texto "Detección completada"
                        content=ft.Text(
                            "¡Detección completada!", 
                            size=25, 
                            weight=ft.FontWeight.BOLD, 
                            color="#3D1E10", 
                            text_align=ft.TextAlign.CENTER
                        ),
                        margin=ft.margin.only(top=30) 
                    ),
                    image_container,
                    result_text_container,
                    ft.Container(expand=True),
                    back_button
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True,
                scroll=ft.ScrollMode.ADAPTIVE
            ),
            expand=True
        )
    )

def take_picture(e, page):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.save("captured_image.jpg")
        cap.release()
        results = predict_disease("captured_image.jpg")
        show_results_page(page, "captured_image.jpg", results) # Pasar los resultados a show_results_page
    else:
        cap.release()
print("No se pudo capturar la imagen")

def upload_image(e, page, file_picker):
    file_picker.pick_files()

def process_upload(image_path, page): # Modificar process_upload
    results = predict_disease(image_path)
    show_results_page(page, image_path, results) # Pasar los resultados a show_results_page

def show_main_page(page):
    page.clean()  
    
    # Crear el FilePicker
    file_picker = ft.FilePicker(on_result=lambda e: process_upload(e.files[0].path, page)) # Modificar FilePicker
    page.overlay.append(file_picker)

    content = ft.Column(
        controls=[
            ft.Row(
                controls=[ft.Container(expand=True), ft.IconButton(icon=ft.icons.HELP_OUTLINE, icon_color="#3D1E10", tooltip="Ayuda")]
            ),
            ft.Row(controls=[ft.Image(src="C:/Users/User/Documents/Proyectos/Proyecto IA/Cacao_app/apk2/assets/logo.png", width=150, height=150)], alignment=ft.MainAxisAlignment.CENTER),
            ft.Text("Detección de Enfermedades del Cacao", size=30, weight=ft.FontWeight.BOLD, color="#3D1E10", text_align=ft.TextAlign.CENTER),
            ft.Container( # Agregar margen vertical a la fila de botones
                margin=ft.margin.symmetric(vertical=5),
                content=ft.Row(
                    expand=True,
                    controls=[
                        ft.Container(
                            expand=True, 
                            content=ft.Column(
                                controls=[
                                    ft.Container(
                                        content=ft.Image(src="C:/Users/User/Documents/Proyectos/Proyecto IA/Cacao_app/apk2/assets/camara-fotografica.png", width=50, height=50),
                                        padding=ft.padding.only(top=10)
                                    ),
                                    ft.Container(
                                        content=ft.Text("CAPTURAR", color="#FFE4C4", weight=ft.FontWeight.BOLD),
                                        padding=ft.padding.only(bottom=5)
                                    ),
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER
                            ), 
                            bgcolor="#3D1E10", 
                            border_radius=10, 
                            padding=ft.padding.symmetric(vertical=15), 
                            alignment=ft.alignment.center, 
                            on_click=lambda e: take_picture(e, page)
                        ),
                        ft.Container(
                            expand=True, 
                            content=ft.Column(
                                controls=[
                                    ft.Container(
                                        content=ft.Image(src="C:/Users/User/Documents/Proyectos/Proyecto IA/Cacao_app/apk2/assets/subir.png", width=50, height=50),
                                        padding=ft.padding.only(top=12)
                                    ),
                                    ft.Container(
                                        content=ft.Text("EXPLORAR", color="#FFE4C4", weight=ft.FontWeight.BOLD),
                                        padding=ft.padding.only(bottom=5)
                                        ),
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.CENTER
                            ), 
                            bgcolor="#3D1E10", 
                            border_radius=12, 
                            padding=ft.padding.symmetric(vertical=15), 
                            alignment=ft.alignment.center, 
                            on_click=lambda e: upload_image(e, page, file_picker)
                        )
                    ]
                )
            )
        ]
    )

    safe_area = ft.SafeArea(
        content=ft.Column(
            controls=[
                content,
                ft.Row(
                    controls=[ft.Text("© The Creators 2025", color="#3D1E10", size=12)],
                    alignment=ft.MainAxisAlignment.CENTER
                )
            ],
            expand=True
        )
    )
    
    page.add(safe_area)

def main(page: ft.Page):
    page.title = "Detección de Enfermedades del Cacao"
    page.bgcolor = "#FFE4C4"
    page.padding = 20
    page.scroll = "adaptive"
    page.theme = ft.Theme(font_family="C:/Users/User/Documents/Proyectos/Proyecto IA/Cacao_app/apk/assets/Rubik-VariableFont_wght.ttf")
    show_main_page(page)  # Mostrar la página principal al inicio

ft.app(target=main)