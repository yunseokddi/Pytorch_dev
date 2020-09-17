from torchsummary import summary
from efficientnet_pytorch import EfficientNet


model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=15)

summary(model, input_size=(3,128,128), device='cpu')