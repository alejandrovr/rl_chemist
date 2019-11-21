import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN_cartpole_original(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        debug = False
        if debug:
            import numpy as np
            #x is torch.float32
            ccc = x.cpu().detach().numpy()
            import matplotlib.pyplot as plt
            if ccc.shape[0]>1:
                for debudidx in range(ccc.shape[0]):
                    image = ccc[debudidx] #first item from batch
                    imaget = image.transpose(2,1,0) #make color channel last
                    plt.figure()
                    plt.imshow(imaget[:,:,:])
                    input('done?')

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))



class XRayMan(nn.Module):
    #just use AlexNet
    def __init__(self, activation='relu'):
        super(XRayMan, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=1),
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

if __name__ == "__main__":
    import numpy as np
    net = XRayMan()
    fake_input = np.random.rand(32, 3, 224, 224) #5 batches, 2channels, 24 box
    fake_input = torch.from_numpy(fake_input).float()
    yhat = net.forward(fake_input)
    print('Done',yhat)


