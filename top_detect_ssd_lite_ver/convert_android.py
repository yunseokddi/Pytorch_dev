import torch
import torchvision

from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

label_path = './models/voc-model-labels.txt'

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
params = torch.load("./models/top_weight3/mb2-ssd-lite-Epoch-499-Loss-1.6979544162750244.pth", map_location = "cpu")

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load_state_dict(params)
net = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

example = torch.rand(1, 3, 720, 1080)

traced_script_module = torch.jit.trace(net.predict(example, 10, 0.4))
traced_script_module.save("./android_ssd_weight.pth")