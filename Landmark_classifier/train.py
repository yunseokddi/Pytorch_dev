import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time

from torchvision import transforms
from PIL import Image
from DataLoader import LandmarkDataset
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import StepLR
from utils import AverageMeter, gap
from tensorboardX import SummaryWriter


def train(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, optimizer: optim,
          scheduler: optim.lr_scheduler, device: torch.device, args=None):

    best_acc = 0.0
    losses = AverageMeter()
    scores = AverageMeter()
    corrects = AverageMeter()

    model.train()
    since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs)

            loss = criterion(output, targets)
            confs, preds = torch.max(output.detach(), dim=1)

            losses.update(loss.item(), inputs.size(0))
            scores.update(gap(preds, confs, targets), inputs.size(0))
            corrects.update((preds == targets).float().sum(), inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            summary.add_scalar('loss', losses.avg, epoch)
            summary.add_scalar('acc', corrects.avg, epoch)
            summary.add_scalar('GAP', scores.avg, epoch)

            if best_acc < corrects.avg:
                best_acc = corrects.avg
                best_model_weights = copy.deepcopy(model.state_dict())

        print(f'epoch: {epoch} Loss: {losses.avg:.4f}' \
              f' Acc: {corrects.avg:.4f} GAP: {scores.avg:.4f}')

        corrects.reset()
        scores.reset()
        losses.reset()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    torch.save(best_model_weights, args.weights_dir + 'best_weights_b0_class_1459.pth')


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

    parser.add_argument('--image_size', dest='image_size', type=int, default=224)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.00001)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)

    parser.add_argument('--weights_dir', dest='weights_dir', default='./checkpoint/')

    args = parser.parse_args()

    batch_size = args.batch_size
    num_epoch = args.epochs
    lr = args.learning_rate
    wd = args.weight_decay
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    summary = SummaryWriter('./runs/')

    train_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomResizedCrop(224),
            transforms.RandomAffine(
                degrees=15, translate=(0.2, 0.2),
                scale=(0.8, 1.2), shear=15, resample=Image.BILINEAR)
        ]),
        transforms.ToTensor(),
        transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),
    ])

    train_dataset = LandmarkDataset('train', train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1049)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    train(model=model, data_loader=train_dataloader, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
          device=device, args=args)
