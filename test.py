import torch
from torchvision import datasets,transforms,models
from model import resnet18
from train import device
from torch.utils.data import Dataset,DataLoader,TensorDataset
from mydataset import *

trainset=datasets.ImageFolder(r"E:\Code\Experiment\Data\whu-laser-2021.1.28\Demos\Test",
                              transform=transforms.ToTensor())


testloader = DataLoader(trainset,
                         batch_size=1,
                         shuffle=False,)
if __name__ == '__main__':
    resnet = models.resnet34()
    resnet.load_state_dict(torch.load("best_resnet50.pkl"))
    resnet = resnet.to(device)
    resnet.eval()
    with torch.no_grad():
        correct_num = 0
        total_num = 0
        batch_num = 0
        #  测试集一共10000张图片，batch_size=30 ,所以一共有334个batch
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total_num += labels.size(0)
            batch_num += 1
            correct_num += (predicted == labels).sum()
            print(predicted==labels)
        print('准确率为:' , (100 * correct_num) / total_num)
        print(correct_num)
        print(type(correct_num))
        print(total_num)