from torchsummary import summary
from efficientnet_pytorch import EfficientNet


model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)

summary(model, input_size=(3,128,128), device='cpu')