import torch
from resnet import ResNet18

device = torch.device('cpu')
PATH = './weights/model_keypoints_68pts_iter_450.pt'


net = ResNet18(136).to(device)
net.load_state_dict(torch.load(PATH, map_location=device))
net.eval()


example = torch.rand(1,3,224,224)

traced_script_module = torch.jit.trace(net, example)
traced_script_module.save("./android_weights/face_landmark_64.pt")
