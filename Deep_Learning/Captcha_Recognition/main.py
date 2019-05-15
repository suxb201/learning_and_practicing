import torch
import torch.nn as nn
import numpy
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BATCH_SIZE = 10


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.labels = numpy.loadtxt(path + 'label.txt')

    def __getitem__(self, index):
        image = Image.open(self.path + str(index) + '.png')
        image = torchvision.transforms.ToTensor()(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return self.labels.shape[0]


train_data = MyDataset('./data/train/')
test_data = MyDataset('./data/test/')

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)

train_datasize = len(train_data)
test_datasize = len(test_data)


# Conv network
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(64 * 22 * 9, 500)
        self.fc2 = nn.Linear(500, 40)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # reshape to (batch_size, 64 * 7 * 30)
        output = self.fc1(x)
        output = self.fc2(output)
        return output


# Train the net
class NCrossEntropyLoss(torch.nn.Module):

    def __init__(self, n=4):
        super(NCrossEntropyLoss, self).__init__()
        self.n = n
        self.total_loss = 0
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, label):
        output_t = output[:, 0:10]
        label = Variable(torch.LongTensor(label.data.cpu().numpy()))
        label_t = label[:, 0]

        for i in range(1, self.n):
            output_t = torch.cat((output_t, output[:, 10 * i:10 * i + 10]),
                                 0)  # 损失的思路是将一张图平均剪切为4张小图即4个多分类，然后再用多分类交叉熵方损失
            label_t = torch.cat((label_t, label[:, i]), 0)
            self.total_loss = self.loss(output_t, label_t)

        return self.total_loss


cnn = ConvNet()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)
loss_func = NCrossEntropyLoss()


def train():
    # cnn.load_state_dict(torch.load('params.pkl'))
    for epoch in range(2):
        for step, (inputs, label) in enumerate(train_dataloader):

            output = cnn(inputs)
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(epoch, step, loss.item())
    torch.save(cnn.state_dict(), 'params.pkl')


def test():
    cnn.load_state_dict(torch.load('params.pkl'))
    tot = 0
    for step, (input_, label_) in enumerate(test_dataloader):
        output_ = cnn(input_)[0]
        ans = ""
        for i in range(4):
            x = torch.max(output_[i * 10:i * 10 + 10], 0)[1]
            ans = ans + str(x.item())
        label_ = ''.join(str(x.item()) for x in label_[0].int())
        if ans != label_:  # wrong guess
            tot = tot + 1
            print(ans, label_)

    print(tot, test_datasize, "wrong", tot / test_datasize)


if __name__ == '__main__':
    train()
    test()
