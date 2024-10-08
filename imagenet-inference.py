import sys
import numpy as np
import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F

import geffnet

import cv2
import numpy as np
import PIL
from PIL import Image

from torchvision import datasets, transforms, models
from tqdm import tqdm
from torchinfo import summary
from fvcore.nn.flop_count import flop_count
from fvcore.nn.activation_count import activation_count
from argparse import Namespace

sys.path.append('/mnt/terabyte/pmousoul_data/Repos/pytorch_files/peleenet')
sys.path.append('/mnt/terabyte/pmousoul_data/Repos/pytorch_files/condensenet')
sys.path.append('/mnt/terabyte/pmousoul_data/Repos/pytorch_files/shufflenetv2')

from peleenet import PeleeNet
from condensenet_converted import CondenseNet
from network import ShuffleNetV2
from pytorchcv.model_provider import get_model as ptcv_get_model


class OpencvResize(object):

  def __init__(self, size=256):
    self.size = size

  def __call__(self, img):
    assert isinstance(img, PIL.Image.Image)
    img = np.asarray(img) # (H,W,3) RGB
    img = img[:,:,::-1] # 2 BGR
    img = np.ascontiguousarray(img)
    H, W, _ = img.shape
    target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = img[:,:,::-1] # 2 RGB
    img = np.ascontiguousarray(img)
    img = Image.fromarray(img)
    return img

class ToBGRTensor(object):

  def __call__(self, img):
    assert isinstance(img, (np.ndarray, PIL.Image.Image))
    if isinstance(img, PIL.Image.Image):
        img = np.asarray(img)
    img = img[:,:,::-1] # 2 BGR
    img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    return img


# Function to calculate top-1 and top-5 accuracy
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
  """
  Computes the accuracy over the k top predictions for the specified values of k
  In top-5 accuracy you give yourself credit for having the right answer
  if the right answer appears in your top five guesses.

  ref:
  - https://pytorch.org/docs/stable/generated/torch.topk.html
  - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
  - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
  - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
  - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

  :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
  :param target: target is the truth
  :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
  e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
  So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
  but if it were either cat or dog you'd accumulate +1 for that example.
  :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
  """
  with torch.no_grad():
      # ---- get the topk most likely labels according to your model
      # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
      maxk = max(topk)  # max number labels we will consider in the right choices for out model
      batch_size = target.size(0)

      # get top maxk indicies that correspond to the most likely probability scores
      # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
      _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
      y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

      # - get the credit for each example if the models predictions is in maxk values (main crux of code)
      # for any example, the model will get credit if it's prediction matches the ground truth
      # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
      # if the k'th top answer of the model matches the truth we get 1.
      # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
      target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
      # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
      correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
      # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

      # -- get topk accuracy
      list_topk_accs = []  # idx is topk1, topk2, ... etc
      for k in topk:
          # get tensor of which topk answer was right
          ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
          # flatten it to help compute if we got it correct for each example in batch
          flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
          # get if we got it right for any of our top k prediction for each example in batch
          tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
          # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
          topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
          list_topk_accs.append(topk_acc)
      return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

# Function to count trainable parameters function
def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()

# Function to get layer output sizes
def get_tensor_dimensions_impl(model, layer, image_size, for_input=False):
  t_dims = None
  def _local_hook(_, _input, _output):
    nonlocal t_dims
    t_dims = _input[0].size() if for_input else _output.size()
    return _output    
  layer.register_forward_hook(_local_hook)
  dummy_var = torch.zeros(1, 3, image_size, image_size).to(device)
  model(dummy_var)
  return t_dims


# Print arguments and models
print("Usage:")
print("python imagenet-inference.py <input-size> <model> <to-onnx> <onnx-name> <model-summary> <activation-size> <accuracy-speed> <write-parameters>")
print("Example:")
print("python imagenet-inference.py 224 squeezenet1_1 0 0 1 1 1 0")
print("Currently the following models are supported:")
print("squeezenet1_1, condensenet_8, mobilenet_v2, squeezenext, peleenet, shufflenet_v2_x1_5, efficientnet-lite0")
if len(sys.argv) != 9:
  print("Invalid usage.")
  print("Program will terminate.")
  exit()


# Select GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the input
input_sz = int(sys.argv[1])
dummy_input = torch.randn(1, 3, input_sz, input_sz)
dummy_input = dummy_input.to(device)


# Input image pre-processing
if not sys.argv[2] == "shufflenet_v2_x1_5":
  transform = transforms.Compose([  
    transforms.Resize(256),         
    transforms.CenterCrop(input_sz),
    transforms.ToTensor(),        
    transforms.Normalize(           
      mean=[0.485, 0.456, 0.406],     
      std=[0.229, 0.224, 0.225]),
  ])
else:
  transform = transforms.Compose([
    OpencvResize(256),
    transforms.CenterCrop(224),
    ToBGRTensor(),
  ])

# Load the data
imagenet_val = datasets.ImageFolder('/mnt/terabyte/datasets/ImageNet/val/' , transform=transform)
val_loader = torch.utils.data.DataLoader(imagenet_val, batch_size=1, shuffle=False)


# Load the pretrained network
if sys.argv[2] == "squeezenet1_1":
  model = models.squeezenet1_1(pretrained=True)

elif sys.argv[2] == "condensenet_8":
  args = Namespace(stages=[4, 6, 8, 10, 8], bottleneck=4, group_1x1=8, group_3x3=8, condense_factor=8, growth=[8, 16, 32, 64, 128], reduction=0.5, num_classes=1000)
  model = CondenseNet(args)
  model = torch.nn.DataParallel(model).cuda()
  checkpoint = torch.load('./pretrained/converted_condensenet_8.pth.tar')
  model.load_state_dict(checkpoint['state_dict'])

elif sys.argv[2] == "mobilenet_v2":
  model = models.mobilenet_v2(pretrained=True)

elif sys.argv[2] == "squeezenext":
  model = ptcv_get_model("sqnxt23v5_w2", pretrained=True)

elif sys.argv[2] == "peleenet":
  model = PeleeNet(num_classes=1000)
  model = torch.nn.DataParallel(model).cuda()
  checkpoint = torch.load('./pretrained/peleenet_acc7208.pth.tar')
  model.load_state_dict(checkpoint['state_dict'])

elif sys.argv[2] == "shufflenet_v2_x1_5":
  args = Namespace(n_class=1000, model_size='1.5x')
  model = ShuffleNetV2(args)
  model = torch.nn.DataParallel(model).cuda()
  checkpoint = torch.load('./pretrained/ShuffleNetV2.1.5x.pth.tar')
  model.load_state_dict(checkpoint['state_dict'])

elif sys.argv[2] == "efficientnet-lite0":
  model = geffnet.create_model('efficientnet_lite0', pretrained=True)

else:
  print("Invalid model selected.")
  print("Program will terminate.")
  exit()
model.eval()
model.to(device)


# Convert to onnx format
if int(sys.argv[3]):
  output_name = sys.argv[4]
  input_names = [ "actual_input" ]
  output_names = [ "output" ]
  torch.onnx.export(model.module,  # use of 'model.module' instead of 'model' because of the 'torch.nn.DataParallel(model)' above
    dummy_input,
    "./onnx/" + output_name,
    opset_version=9,
    verbose=False,
    input_names=input_names,
    output_names=output_names,
    export_params=True)


# Print model and its summary
if int(sys.argv[5]):
  # print(model)
  summary(model,
    verbose=1,
    depth=7,
    input_size=(1,3,input_sz,input_sz),
    col_names=["kernel_size", "input_size", "output_size", "num_params", "mult_adds"])
  print("FB: Trainable parameters(M):", params_count(model)/(10**6))
  gflop_dict, _ = flop_count(model, dummy_input)
  gflops = sum(gflop_dict.values())
  print("FB: MACC count(M):", gflops*1000)
  activation_dict, _ = activation_count(model, dummy_input)
  activation = sum(activation_dict.values())
  print("FB: Activation count(M):", activation)



# Print total activation size
if int(sys.argv[6]):
  activation_sz = 0
  for name, layer in model.named_modules():
    #if isinstance(layer, torch.nn.Conv2d):
    if not get_tensor_dimensions_impl(model, layer, input_sz)==None:
      activation_sz += get_tensor_dimensions_impl(model, layer, input_sz).numel()
  print("TOTAL: Activation size(M):", activation_sz/1000000)


# Benchmark top-1 and top-5 accuracy
if int(sys.argv[7]):
  correct1 = 0
  correct5 = 0
  total = len(imagenet_val)
  print("Total validation images:", total)

  with torch.no_grad():
    # Iterate through test set minibatchs 
    for images, labels in tqdm(val_loader):
      images, labels = images.to(device), labels.to(device)

      # Forward pass
      x = images
      y = model(x)

      list = accuracy(y, labels, topk=(1,5))
      correct1 += float(list[0])
      correct5 += float(list[1])
   
  print('Top-1 accuracy: {}'.format(correct1/total))
  print('Top-5 accuracy: {}'.format(correct5/total))


# Write parameters to file
if int(sys.argv[8]):
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(param.data.size())
#     print(name, param.data)
