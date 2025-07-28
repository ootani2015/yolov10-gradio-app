import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image

# ローカルのYOLOv10モデルを読み込み
model = torch.load("yolov10s.pt", map_location=torch.device("cpu"))
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
    description="ローカルのYOLOv10モデルを使って画像から物体検出します。"
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8080)
