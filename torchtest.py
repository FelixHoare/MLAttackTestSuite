import torch
from torchvision import models

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")
    
# Create a dummy input with your custom size (batch_size=1, channels=3, height=200, width=200)
dummy_input = torch.randn(1, 3, 200, 200)

# Load the VGG16 model
base_model = models.vgg16(weights='IMAGENET1K_V1')

# Pass the dummy input through the feature extractor
output_features = base_model.features(dummy_input)

# Check the output shape
print(output_features.shape) # Should print [1, 512, 6, 6]