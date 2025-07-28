import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image

# YOLOv10モデルの読み込み
model = torch.hub.load('WongKinYiu/yolov10', 'yolov10s', trust_repo=True)
model.eval()

def detect_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = model(img)
    rendered_img = results.render()[0]
    rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rendered_img)

app = gr.Interface(
    fn=detect_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="YOLOv10 物体検知アプリ",
    description="画像をアップロードすると、YOLOv10で物体検出を行います"
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8080)
