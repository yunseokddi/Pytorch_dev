import torch
import torchvision

from net import Dc_model
from torchvision.models import resnet18

device = torch.device('cpu')

weight_PATH = './weights/eyes_crop_new_ver_epoch99.pth'

model = resnet18(pretrained=True).to(device)
model_ = Dc_model().to(device)
model.fc = model_

model.load_state_dict(torch.load(weight_PATH, map_location=device))

example = torch.rand(1,3,50,50)

trace_script_module = torch.jit.trace(model, example)
trace_script_module.save('./android_weights/eye_close_resnet18.pt')