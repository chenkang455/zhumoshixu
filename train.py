import torch
import torch.nn as nn
from torchvision import datasets,transforms
from torchvision import models
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset=datasets.ImageFolder(r"/content/drive/MyDrive/Colab Notebooks/Train-ResNet",
                              transform=transforms.ToTensor())
testset=datasets.ImageFolder(r"/content/drive/MyDrive/Colab Notebooks/Test",
                              transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=True,num_workers=2,pin_memory=True
    )  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
testloader = torch.utils.data.DataLoader(
    testset, batch_size=1, shuffle=False,pin_memory=True
    )  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取


classes = ('PenFirstInk', 'PenFirstRed')
PATH = 'best_resnet34.pkl'
net=models.resnet34()
net.load_state_dict(torch.load(PATH, map_location='cpu'))
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)  # 选好优化方式

# net.load_state_dict(torch.load("resnet50-0676ba61.pth"))
net = net.to(device)
# 定义损失函数和优化方式
loss_fn = nn.CrossEntropyLoss().to(device)  # 损失函数为交叉熵，多用于多分类问题

# 训练
if __name__ == "__main__":
    losses=[]
    rate=0
    max_rate=0
    for epoch in range(10):
        total_loss=0
        net.train()
        # 训练共50000张图片， batch_size=30, 每个batch有1667个数据
        for batchidx, (x, labels) in enumerate(trainloader):
            x, labels = x.to(device), labels.to(device)
            x = net.forward(x)
            loss = loss_fn(x, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=float(loss)
            print(1)
        losses.append(total_loss)
        print('这是第{}迭代，loss是{}'.format(epoch+1, total_loss))
        net.eval()
        with torch.no_grad():
            correct_num = 0
            total_num = 0
            batch_num = 0
            #  测试集一共10000张图片，batch_size=30 ,所以一共有334个batch
            for images, labels in testloader:
              images, labels = images.to(device), labels.to(device)
              outputs = net(images)
              _, predicted = torch.max(outputs.data, 1)
              total_num += labels.size(0)
              batch_num += 1
              correct_num += (predicted == labels).sum()
            rate= (100 * correct_num) / total_num
            if(rate>max_rate):
              torch.save(net.state_dict(),"best_resnet34_version2.pkl")
            print('准确率为:', rate)
    print(losses)
    torch.save(net.state_dict(), "last_resnet50_version2.pkl")
    plt.plot(losses)
    plt.show()

