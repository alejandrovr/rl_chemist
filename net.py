import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class XRayMan(nn.Module):
    #just use AlexNet
    def __init__(self, activation='relu'):
        super(XRayMan, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=1),
            self.activation,
            nn.MaxPool2d(kernel_size=3),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=1),
            self.activation,
            nn.MaxPool2d(kernel_size=2),
        )        
        
        self.wrap_up = nn.Sequential(
            nn.Linear(39200, 512),
            self.activation,
            nn.Linear(512, 3),
        )

    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.wrap_up(out)
        #print(out.shape)
        return out



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) #outputs: 200, 20, 1, 6 (b,f,h,w) #120 fingerprint
        #each frame (timestep) is "fingerprinted" in 120 bits by the CNN filters
        print('after CNN:',x.shape)
        x = x.view(-1, 320) #just using it to extract features
        return x


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size=120, 
            hidden_size=64, 
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(64,10)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size() #50, 4, 1, 7, 28
        c_in = x.view(batch_size * timesteps, C, H, W) #200, 1, 7, 28
        c_out = self.cnn(c_in) #(75, 320)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])
        
        return F.log_softmax(r_out2, dim=1)



if __name__ == "__main__":
    import numpy as np
    net = XRayMan()
    fake_input = np.random.rand(32, 3, 224, 224) #5 batches, 2channels, 24 box
    fake_input = torch.from_numpy(fake_input).float()
    yhat = net.forward(fake_input)
    print('Done',yhat)


