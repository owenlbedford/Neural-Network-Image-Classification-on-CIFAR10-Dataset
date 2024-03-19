import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim

from IPython import display
from torch import nn

# Read training and test data

# define transforms for data augmentation
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# load the CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

batch_size = 4

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cpu")


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.l1 = nn.Linear(3, 5)
        self.l2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

        self.conv1b1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2b1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv3b1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv4b1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv5b1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        self.conv1b2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv2b2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3b2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.sequential1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.sequential2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        # Creates 5 convolutional layers, where k = 5 for block 1
        conv1b1out = self.conv1b1(x)
        conv2b1out = self.conv2b1(x)
        conv3b1out = self.conv3b1(x)
        conv4b1out = self.conv4b1(x)
        conv5b1out = self.conv5b1(x)

        # Calculuates the spatial average per channel.
        spatialAverage = x.mean(dim=(2, 3))

        a = self.l1(spatialAverage)
        # Passes a through a non-linear activation function in order to have a in the form g(SpatialAveragePool(X)W), where g is the non-linear action function
        a = self.relu(a)

        # Creates copies to ensure consistent dimensions
        conv1b1out_copy = conv1b1out.clone()
        conv2b1out_copy = conv2b1out.clone()
        conv3b1out_copy = conv3b1out.clone()
        conv4b1out_copy = conv4b1out.clone()
        conv5b1out_copy = conv5b1out.clone()

        # Multiplies each conv layer by its correspond a value, for example a1conv1b1out, a2conv2b1out, etc.
        for m in range(0, batch_size):
            conv1b1out_copy[m, :, :, :] = torch.mul(a[m][0], conv1b1out[m, :, :, :])
            conv2b1out_copy[m, :, :, :] = torch.mul(a[m][1], conv2b1out[m, :, :, :])
            conv3b1out_copy[m, :, :, :] = torch.mul(a[m][2], conv3b1out[m, :, :, :])
            conv4b1out_copy[m, :, :, :] = torch.mul(a[m][3], conv4b1out[m, :, :, :])
            conv5b1out_copy[m, :, :, :] = torch.mul(a[m][4], conv5b1out[m, :, :, :])

        # This section of code adds the conv layers to put o in the form o=a1conv1b1out(x) + ... + a5conv5b1out(x).
        o = torch.add(conv1b1out_copy, conv2b1out_copy)
        o = torch.add(o, conv3b1out_copy)
        o = torch.add(o, conv4b1out_copy)
        o = torch.add(o, conv5b1out_copy)

        # Sequential which applies the components learnt from week 5 - 8.
        o = self.sequential1(o)

        # Block 2
        # Creates 3 convolutional layers, where k = 3 for block 2
        conv1b2out = self.conv1b2(o)
        conv2b2out = self.conv2b2(o)
        conv3b2out = self.conv3b2(o)

        # Calculuates the spatial average per channel.
        spatialAverage = o.mean(dim=(2, 3))
        a = self.l2(spatialAverage)
        # Passes a through a non-linear activation function in order to have a in the form g(SpatialAveragePool(X)W), where g is non-linear action function
        a = self.relu(a)

        # Creates copies to ensure consistent dimensions
        conv1b2out_copy = conv1b2out.clone()
        conv2b2out_copy = conv2b2out.clone()
        conv3b2out_copy = conv3b2out.clone()

        # Multiplies each conv layer by its correspond a value, for example a1conv1b2out, a2conv2b2out.
        for m in range(0, batch_size):
            conv1b2out_copy[m, :, :, :] = torch.mul(a[m][0], conv1b2out[m, :, :, :])
            conv2b2out_copy[m, :, :, :] = torch.mul(a[m][1], conv2b2out[m, :, :, :])
            conv3b2out_copy[m, :, :, :] = torch.mul(a[m][2], conv3b2out[m, :, :, :])

        # This section of code adds the conv layers to put o in the form o=a1conv1b2out(x) + a2conv2b2out(x) + a3conv3b2out(x).
        o = torch.add(conv1b2out_copy, conv2b2out_copy)
        o = torch.add(o, conv3b2out_copy)
        # Sequential which applies the components learnt from week 5 - 8.
        o = self.sequential2(o)

        # Calculuates the spatial average per channel for the output of the last block (which is block 2)
        f = o.mean(dim=(2, 3))
        # Passes f to softmax regression classifer
        finalOutput = self.fc1(f)
        return finalOutput


network = Net()
network.to(device)
# Cross Entropy loss
criterion = nn.CrossEntropyLoss()
# Adam Optimizer with learning rate of 0.001
optimizer = optim.Adam(network.parameters(), lr=0.001)


# Defines the training loop
def train(net, train_loader, lossf, opt):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        opt.zero_grad()
        outputs = net(inputs)
        loss = lossf(outputs, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * (correct / total)
    return epoch_loss, epoch_acc


# Defines the testing loop
def test(net, test_loader, lossf):
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = lossf(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * (correct / total)
    return epoch_loss, epoch_acc


# Define the number of epochs
num_epochs = 70

# Train the network
for epoch in range(num_epochs):
    train_loss, train_acc = train(network, trainloader, criterion, optimizer)
    test_loss, test_acc = test(network, testloader, criterion)
    print(
        f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

corr = 0
tot = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for dat in testloader:
        images, label = dat[0].to(device), dat[1].to(device)
        # calculate outputs by running images through the network
        output = network(images)
        # the class with the highest energy is what we choose as prediction
        _, predict = torch.max(output.data, 1)
        tot += label.size(0)
        corr += (predict == label).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * corr // tot} %')
