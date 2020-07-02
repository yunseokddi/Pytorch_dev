import torch
from resnet import ResNet18

PATH = './weights/model_keypoints_68pts_iter_450.pt'


net = ResNet18(136).to('cpu')
net.load_state_dict(torch.load(PATH))
net.eval()


example = torch.rand(1,3,250,250)

traced_script_module = torch.jit.trace(net, example)
traced_script_module.save("./weight_for_android.pt")

print(net)