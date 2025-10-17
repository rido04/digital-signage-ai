import torch
import torch_directml

# Test DirectML
device = torch_directml.device()
print(f"✅ DirectML device: {device}")

# Test tensor operations
x = torch.randn(100, 100).to(device)
y = torch.randn(100, 100).to(device)
z = x @ y
print(f"✅ Matrix multiplication successful: {z.shape}")

print("\n🎉 DirectML is working!")