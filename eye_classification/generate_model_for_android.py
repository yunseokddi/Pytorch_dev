import torch
from model import Net

PATH = './weights/trained.pth'

model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()

example = torch.rand(1,1,26,34)

traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("./weights/weight_for_android.pt")

print(model)