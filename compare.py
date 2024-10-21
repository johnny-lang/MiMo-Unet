import os
import torch
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from utils import Adder
from models.MIMOUNet import build_net  # Giả định bạn có hàm này để xây dựng mô hình


def load_model(model_path):
    model = build_net('MIMO-UNet')  # Hoặc 'MIMO-UNetPlus'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])
    model.eval()
    return model


def process_single_image(model, image_path, result_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Kiểm tra xem tệp có tồn tại không
    if not os.path.exists(image_path):
        print(f"Error: The image path {image_path} does not exist.")
        return

    # Tải và chuẩn bị ảnh
    input_img = Image.open(image_path).convert('RGB')
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
    save_name = os.path.join(result_dir, os.path.basename(image_path))
    pred_image.save(save_name)
    print(f'Saved output image to {save_name}')

    # Chuyển đổi ảnh đầu vào sang numpy và chuẩn hóa
    input_numpy = np.array(input_img) / 255.0  # Chuyển đổi giá trị 0-255 sang 0-1
    pred_numpy = np.array(pred_image) / 255.0  # Chuyển đổi giá trị 0-255 sang 0-1

    # In kích thước của ảnh đầu vào và đầu ra
    print(f'Input image size: {input_numpy.shape}, Output image size: {pred_numpy.shape}')

    # Tính PSNR
    psnr = peak_signal_noise_ratio(input_numpy, pred_numpy, data_range=1)
    print(f'PSNR: {psnr:.2f} dB')




if __name__ == "__main__":
    model_path = 'results/MIMO-UNet/weights/model.pkl'
    input_image_path = 'dataset/GOPRO_Large/test/GOPR0384_11_05/blur_gamma/004090.png'  # Đường dẫn đến ảnh đầu vào
    result_dir = 'results'

    # Tải mô hình
    model = load_model(model_path)

    # Xử lý ảnh đơn lẻ
    process_single_image(model, input_image_path, result_dir)



