import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet

df = pd.read_excel('./data/KFace_data_information_Folder1_400.xlsx')

age_dict = {'20대': '0',
            '30대': '1',
            '40대': '2',
            '50대': '3'}

sex_dict = {'남': '0',
            '여': '1'}

classes_dict = {'00': 0,  # 20대 남자
                '01': 1,  # 20대 여자
                '10': 2,  # 30대 남자
                '11': 3,  # 30대 여자
                '20': 4,  # 40대 남자
                '21': 5,  # 40대 여자
                '30': 6,  # 50대 남자
                '31': 7}  # 50대 여자

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomChoice([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    ]),
    transforms.ToTensor(),
    transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),
])

train_dataset = ImageFolder(root='./sample_data/image', transform=train_transforms)  # change image directory
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=8)
