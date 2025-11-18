import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform) # default = True
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# # show some of the training images
# import matplotlib.pyplot as plt
# import numpy as np
# #import sys, os

# # functions to show an image
# def imgshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# if __name__ ==  '__main__':
#     # get some random training images
#     dataiter = iter(trainloader)
#     images, labels = dataiter.next()
#     # show images
#     imgshow(torchvision.utils.make_grid(images))
#     # print labels
#     print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# model structure
import torch.nn as nn
import torch.nn.functional as F

# set CUDA device
use_cuda = 0
device = torch.device("cuda:0" if torch.cuda.is_available()&use_cuda else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

net = Net()
net.to(device)

# define loss
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# define optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
import time
start_time = time.time()
for epoch in range(3):
     training_loss = 0.0
     for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        #zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize/update
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, training_loss / 2000))
            training_loss = 0.0

end_time = time.time()
print ('The time cost for training process is: %7f' % (end_time-start_time))
print('End Training')

print('Start Testing')

correct_total = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
     images, labels = data
     images, labels = images.to(device), labels.to(device)
     outputs = net(images)
     _, predicted = torch.max(outputs.data, 1)
     total += labels.size(0)
     correct_total += (predicted == labels).sum()
     correct_minibatch = (predicted == labels).type(torch.int64)
     for i in range (4):
          label = labels[i]
          class_correct[label] += correct_minibatch[i]
          class_total[label] += 1 

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_total / total))
for i in range(10):
     print ('Accuracy of the class %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
