import os
import torch
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import gradio as gr
from models.MIMOUNet import build_net  # Giả định bạn có hàm này để xây dựng mô hình


def load_model(model_path):
    model = build_net('MIMO-UNet')  # Hoặc 'MIMO-UNetPlus'
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict['model'])
    model.eval()
    return model


def process_single_image(model, input_img, result_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Chuẩn bị ảnh (đã là đối tượng Image)
    input_tensor = F.to_tensor(input_img).unsqueeze(0).to(device)  # Thêm batch dimension

    # Dự đoán
    with torch.no_grad():
        pred = model(input_tensor)

    # Xử lý đầu ra
    pred_clip = torch.clamp(pred[2], 0, 1)
    pred_numpy = pred_clip.squeeze(0).cpu().numpy()

    # Chuyển đổi đầu ra về định dạng ảnh
    pred_image = (pred_numpy * 255).astype(np.uint8)  # Chuyển đổi về giá trị 0-255
    pred_image = Image.fromarray(pred_image.transpose(1, 2, 0), 'RGB')  # Chuyển từ (C, H, W) sang (H, W, C)

    # Resize lại về kích thước gốc
    pred_image = pred_image.resize(input_img.size, Image.BICUBIC)

    # Lưu ảnh
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_name = os.path.join(result_dir, 'output_image.png')
    pred_image.save(save_name)
    print(f'Saved output image to {save_name}')

    return pred_image  # Trả về ảnh đầu ra


def inference(image):
    result_dir = 'results'  # Thay đổi đường dẫn đầu ra nếu cần
    output_image = process_single_image(model, image, result_dir)
    return output_image


# Load your model
model_path = 'results/MIMO-UNet/weights/MIMO-UNet (1).pkl'
model = load_model(model_path)

# Gradio Interface
interface = gr.Interface(
    fn=inference,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=gr.Image(type="pil", label="Output Image"),
    title="Image Deblurring Application",
    description="Upload an image to apply deblurring.",
)

interface.launch()
