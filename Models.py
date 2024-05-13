import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def hardwire_layer(input):
    """
    Proprocess the given consecutive input frames into 5 different styles. 
    input : (N, frames, height, width, colors=3)
    ex) TRECVID dataset : (N, 7, 60, 40, 3),  KTH dataset : (N, 9, 80, 60, 3)
    
    ##################################### hardwired frames #####################################
    # content: [[[gray frames], [grad-x frames], [grad-y frames], [opt-x frames], [opt-y frames]], ...]
    # shape:   [[[---- f ----], [----- f -----], [----- f -----], [---- f-1 ---], [---- f-1 ---]], ...]
    #           => total: 5f-2 frames
    ############################################################################################
    """
    assert len(input.shape) == 5 and input.shape[4] == 3
    print("Before hardwired layer:\t", input.shape)
    N, f, h, w = input.shape[:4] 
    
    hardwired = np.zeros((N, 5*f-2, h, w)) # gray,gradient-x,y for each frame (fx3)  +   optflow-x,y for each two consecutive frames ((f-1)x2)
    gray = np.zeros((N, f, h, w), dtype='uint8')
    for i in range(N):
        for j in range(f):
            # gray
            gray[i,j,:,:] = cv2.cvtColor(input[i,j,:,:,:], cv2.COLOR_BGR2GRAY)
            hardwired[i,0+j,:,:] = gray[i,j,:,:]
            
            # gradient-x, gradient-y
            hardwired[i,f+j,:,:], hardwired[i,2*f+j,:,:] = np.gradient(gray[i,j,:,:], axis=1), np.gradient(gray[i,j,:,:], axis=0)
            
            # optflow-x,y
            if j == f-1: 
                continue
            flow = cv2.calcOpticalFlowFarneback(gray[i,j,:,:],gray[i,j+1,:,:],None,0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            hardwired[i,3*f+j,:,:], hardwired[i,4*f-1+j,:,:] = flow[:,:,0], flow[:,:,1]
    
    hardwired = torch.reshape(torch.from_numpy(hardwired), (N, 1, 5*f-2, h, w))
    print("After hardwired layer :\t", hardwired.shape)
    return hardwired


#* 3D-CNN Model
class Original_Model(nn.Module):
    """
    3D-CNN model designed by the '3D-CNN for HAR' paper. 
    Input Shape: (N, C_in=1, Dimension=5f-2, Height=h, Width=w)
    """
    def __init__(self, verbose=False, input_dim=7):
        """You need to give the dimension (*, *, Dim, *, *) as a parameter."""
        super(Original_Model, self).__init__()
        self.verbose = verbose
        self.f = input_dim
        self.dim = self.f * 5 - 2
        self.dim1, self.dim2 = (self.dim-10)*2, (self.dim-20)*6
        print(self.dim, self.dim1, self.dim2)
        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,7,7), stride=1)
        self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,6), stride=1)
        self.pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.dim1, 3, bias=False)
        self.pool2 = nn.MaxPool2d(3)
        self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=128, kernel_size=(7,4), stride=1)
        self.fc1 = nn.Linear(128, 3, bias=False)
    
    def forward(self, x):
        if self.verbose: print("연산 전:\t", x.size())
        assert x.size()[1] == 1

        (x1, x2, x3, x4, x5) = torch.split(x, [self.f-1,self.f-1,self.f,self.f,self.f], dim=2)
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(self.conv1(x2))
        x3 = F.relu(self.conv1(x3))
        x4 = F.relu(self.conv1(x4))
        x5 = F.relu(self.conv1(x5))
        x = torch.cat([x1, x2, x3, x4, x5], dim=2)
        if self.verbose: print("conv1 연산 후:\t", x.shape)

        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.pool1(x)
        x = x.view(-1, 2, self.dim1//2, x.shape[2], x.shape[3])
        if self.verbose: print("pool1 연산 후:\t", x.shape)

        (x1, x2, x3, x4, x5) = torch.split(x, [self.f-3,self.f-3,self.f-2,self.f-2,self.f-2], dim=2)
        x1 = F.relu(self.conv2(x1))
        x2 = F.relu(self.conv2(x2))
        x3 = F.relu(self.conv2(x3))
        x4 = F.relu(self.conv2(x4))
        x5 = F.relu(self.conv2(x5))
        x = torch.cat([x1, x2, x3, x4, x5], dim=2)
        if self.verbose: print("conv2 연산 후:\t",x.shape)

        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.pool2(x)
        x = x.view(-1, self.dim2, x.shape[2], x.shape[3])
        if self.verbose: print("pool2 연산 후:\t", x.shape)

        x = F.relu(self.conv3(x))
        if self.verbose: print("conv3 연산 후:\t", x.shape)

        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        if self.verbose: print("fc1 연산 후:\t", x.shape)

        return x
