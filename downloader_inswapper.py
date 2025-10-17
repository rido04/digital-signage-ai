import os
import requests
from tqdm import tqdm

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ Downloaded to: {save_path}")

# URL model dari HuggingFace
url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"

# Path tujuan
model_dir = os.path.expanduser("~/.insightface/models")
model_path = os.path.join(model_dir, "inswapper_128.onnx")

print("Downloading inswapper_128.onnx...")
download_file(url, model_path)
print("✅ Download complete!")