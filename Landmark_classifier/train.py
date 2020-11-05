import torch
import argparse
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import copy

from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from dataloader import TrainDataSet, collate_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', dest='train_dir', default="./dataset/train/")
    parser.add_argument('--train_csv_dir', dest='train_csv_dir', default="./dataset/train.csv")
    parser.add_argument('--train_csv_exist_dir', dest='train_csv_exist_dir', default="./dataset/train_exist.csv")

    parser.add_argument('--test_dir', dest='test_dir', default="./dataset/test/")
    parser.add_argument('--test_csv_dir', dest='test_csv_dir', default="./dataset/sample_submission.csv")
    parser.add_argument('--test_csv_exist_dir', dest='test_csv_exist_dir',
                        default="./dataset/sample_submission_exist.csv")

    parser.add_argument('--test_csv_submission_dir', dest='test_csv_submission_dir',
                        default="./dataset/my_submission.csv")
    parser.add_argument('--model_dir', dest='model_dir', default="./checkpoint/")

    parser.add_argument('--image_size', dest='image_size', type=int, default=256)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--wd', dest='wd', type=float, default=1e-5)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)

    parser.add_argument('--train', dest='train', type=bool, default=True)
    parser.add_argument('--load_epoch', dest='load_epoch', type=int, default=29)

    args = parser.parse_args()

    image_size = args.image_size

    data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = TrainDataSet(train_dir=args.train_dir, train_csv_dir=args.train_csv_dir,
                                 train_csv_exist_dir=args.train_csv_exist_dir, transform=data_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=1049)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for iter, (image,label) in enumerate(train_dataloader):
            print(iter)
            print(label)

            print(image)

            optimizer.zero_grad()

            image /= 255.0
            if image.ndimension() == 3:
                image = image.unsqueeze(0)

            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * image.size(0)
            running_corrects += torch.sum(preds == label.data)

            scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))