### YOUR CODE HERE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from tqdm import tqdm, trange

# Count trainable parameters function
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-processing
from torchvision import transforms
transform = transforms.Compose([             #[1]
  transforms.Resize(256),                    #[2]
  transforms.CenterCrop(224),                #[3]
  transforms.ToTensor(),                     #[4]
  transforms.Normalize(                      #[5]
  mean=[0.485, 0.456, 0.406],                #[6]
  std=[0.229, 0.224, 0.225]                  #[7]
  )])

# Load the data
imagenet_val = datasets.ImageFolder('/mnt/terabyte/datasets/ImageNet/val/' , transform=transform)
val_loader = torch.utils.data.DataLoader(imagenet_val, batch_size=100, shuffle=False)
    
# Load the pretrained network
model = models.shufflenet_v2_x1_0(pretrained=True)
model.eval()
model.to(device)
print(model)

# Print parameters
max_param_sz = 0
for name, param in model.named_parameters():
  if param.requires_grad:
    #max_param_sz = max(max_param_sz, param.data.size)
    print(param.data.size())
#       print(name, param.data)
print("Max parameter size:", max_param_sz)
print("Trainable parameters:", count_parameters(model))

## Testing
correct = 0
total = len(imagenet_val)
print("Total validation images:", total)

with torch.no_grad():
  # Iterate through test set minibatchs 
  for images, labels in tqdm(val_loader):
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    x = images
    y = model(x)
    
    predictions = torch.argmax(y, dim=1)
    correct += torch.sum((predictions == labels).float())

    
print('Test accuracy: {}'.format(correct/total))

# Make sure to print out your accuracy on the test set at the end.
