import torch
import torch.nn as nn
from torchvision import transforms, models
import gradio as gr
import cv2

device = torch.device("cpu")
# 构建同样的resnet18模型结构
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
# 加载保存好的模型权重
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

def predict_image(img):
    img = cv2.resize(img, (224, 224))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)
    with torch.no_grad():
        out = model(img_tensor)
        pred_idx = torch.argmax(out).item()
    return "识别结果：这是猫" if pred_idx == 0 else "识别结果：这是狗"

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="上传猫狗图片"),
    outputs="text",
    title="猫狗图像分类预测",
    description="上传一张猫或者狗的图片，模型自动识别"
)
demo.launch()
