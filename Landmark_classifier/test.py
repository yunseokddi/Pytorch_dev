import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torchvision import transforms
from DataLoader import LandmarkDataset
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet


def test(model: nn.Module, data_loader: DataLoader, device: torch.device):
    submission = pd.read_csv('./dataset/sample_submisstion.csv', index_col='id')
    count = 0.
    model.eval()

    for image_id, inputs in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = nn.Softmax(dim=1)(outputs)

        landmark_ids = outputs.argmax(dim=1).cpu().numpy()
        landmark_ids = np.array(landmark_ids, dtype=int)
        confidence = outputs[0, landmark_ids].cpu().detach().numpy()
        submission.loc[image_id, 'landmark_id'] = landmark_ids
        submission.loc[image_id, 'conf'] = confidence

        print('{}% tested'.format(round(count/2372*100, 3)))
        count += 1

    submission.to_csv('submission.csv')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),
                                         ])

    testset = LandmarkDataset('test', test_transform)

    test_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)

    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1049)
    model.load_state_dict(torch.load('./checkpoint/best_weights_b0_class_1459.pth'))
    model.to(device)

    test(model=model, data_loader=test_loader, device=device)
