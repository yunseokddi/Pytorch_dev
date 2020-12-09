import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import argparse

from torchvision.models import resnet101, vgg16_bn
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

transforms = transforms.Compose([
    transforms.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    since = time.time()

    best_acc = 0.0

    for epoch in range(epochs):
        train_running_score = 0.0
        train_running_loss = 0.0

        val_running_score = 0.0
        val_running_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            inputs = data[0].to(device)
            targets = data[1].to(device)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()

            val, index_ = torch.max(pred, axis=1)
            train_running_score += torch.sum(index_ == targets.data).item()
            train_running_loss += loss.item()

            summary.add_scalar('train_loss', loss.detach().item(), epoch)
            summary.add_scalar('train_acc', torch.sum(index_ == targets.data).item(), epoch)

        train_epoch_acc = train_running_score / len(train_loader)
        train_epoch_loss = train_running_loss / len(train_loader)

        print("train epoch [{}] loss: {}, accuracy: {:.4f}".format(epoch, train_epoch_loss, train_epoch_acc))

        model.eval()
        for j, val_data in enumerate(val_loader):
            val_inputs = val_data[0].to(device)
            val_targets = val_data[1].to(device)

            val_pred = model(val_inputs)
            val_loss = criterion(val_pred, val_targets)

            val, index__ = torch.max(val_pred, axis=1)
            val_running_score += torch.sum(index__ == val_targets.data).item()
            val_running_loss += val_loss.item()

            summary.add_scalar('val_loss', val_loss.detach().item(), epoch)
            summary.add_scalar('val_acc', torch.sum(index__ == val_targets.data).item(), epoch)

        val_epoch_acc = val_running_score / len(val_loader)
        val_epoch_loss = val_running_loss / len(val_loader)

        print("val epoch [{}] loss: {}, accuracy: {:.4f}".format(epoch, val_epoch_loss, val_epoch_acc))
        print()

        if best_acc < val_epoch_acc:
            best_acc = val_epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            best_lr = lr

        scheduler.step()

    time_elapsed = time.time() - since

    torch.save(best_model_weights, args.weights_dir + args.model + '_' + str(round(best_acc, 4)) + '_weight.pth')

    return time_elapsed, best_acc, best_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', dest='root_dir', default="./data/2_class_img_data")

    parser.add_argument('--model', dest='model', type=str, default='efficientnet')
    parser.add_argument('--image_size', dest='image_size', type=int, default=128)
    parser.add_argument('--epochs', dest='epochs', type=int, default=30)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)

    parser.add_argument('--weights_dir', dest='weights_dir', default='./checkpoint/')
    parser.add_argument('--runs_dir', dest='runs_dir', default='./runs/')

    args = parser.parse_args()

    epochs = args.epochs
    image_size = args.image_size
    lr = args.learning_rate

    summary = SummaryWriter(args.runs_dir)

    dataset = ImageFolder(root=args.root_dir, transform=transforms)
    train_dataset, val_dataset = random_split(dataset, [6500, 1000])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == 'efficientnet':
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=3)
        model.to(device)

    elif args.model == 'vgg':
        model = vgg16_bn(pretrained=True)

        for param in model.features.parameters():
            param.require_grad = False

        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_features, 3)])  # Add our layer with 4 outputs
        model.classifier = nn.Sequential(*features)

        model.to(device)

    elif args.model == 'resnet':
        model = resnet101(pretrained=True)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)

        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    time_elapsed, best_acc, best_lr = train()

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val lr: {:4f}'.format(best_lr))
