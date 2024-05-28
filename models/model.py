import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

def hardwire_layer(input, device, verbose=False):
    """
    Proprocess the given consecutive input (grayscaled) frames into 5 different styles. 
    input : array of shape (N, frames, height, width)
    ex) TRECVID dataset : (N, 7, 60, 40),  KTH dataset : (N, 9, 80, 60)
    
    ##################################### hardwired frames #####################################
    # content: [[[gray frames], [grad-x frames], [grad-y frames], [opt-x frames], [opt-y frames]], ...]
    # shape:   [[[---- f ----], [----- f -----], [----- f -----], [---- f-1 ---], [---- f-1 ---]], ...]
    #           => total: 5f-2 frames
    ############################################################################################
    """
    assert len(input.shape) == 4 
    if verbose: print("Before hardwired layer:\t", input.shape)
    N, f, h, w = input.shape
    
    hardwired = torch.zeros((N, 5*f-2, h, w)).to(device) 
    input = input.to(device)

    gray = input.clone()
    x_gradient, y_gradient = torch.gradient(gray, dim=[2,3])

    x_flow = []
    y_flow = []
    for i in range(N):
        prevs = input[i, :-1].detach().cpu().numpy()
        nexts = input[i, 1:].detach().cpu().numpy()
        for prev_, next_ in zip(prevs, nexts):
            flow = cv2.calcOpticalFlowFarneback(prev_, next_,
                                                None, 0.5, 3, 15, 3, 5, 1.1,
                                                cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            x_flow.append(torch.tensor(flow[...,0]))
            y_flow.append(torch.tensor(flow[...,1]))
    x_flow = torch.stack(x_flow, dim=0).reshape(N, f-1, h, w).to(device)
    y_flow = torch.stack(y_flow, dim=0).reshape(N, f-1, h, w).to(device)
    
    hardwired = torch.cat([gray, x_gradient, y_gradient, x_flow, y_flow], dim=1)
    hardwired = hardwired.unsqueeze(dim=1)
    if verbose: print("After hardwired layer :\t", hardwired.shape)
    return hardwired


def compute_sift_descriptors(frames, sift=None):
    """
    Extract SIFT descriptors from consecutive frames. 
    Input : frames (shape: [f, h, w])
    Output : descriptors (shape: [num_descriptors, 128]). 
        The number of descriptors depends on the SIFT result, but guarantees to have more than 20. Each descriptor feature has 128 dimension.
    """
    if sift is None :
        sift = cv2.SIFT_create()
    all_descriptors_list = [np.random.rand(20, 128).astype(np.float32)]
    all_descriptors_list += [sift.detectAndCompute(image, None)[1] for image in frames]
    all_descriptors = np.vstack([x for x in all_descriptors_list if x is not None])
    if len(all_descriptors) > 40 : 
        return all_descriptors[20:]
    else :
        return all_descriptors[len(all_descriptors)-20:]

def compute_mehi(frames):
    """
    Compute MEHI (Motion Edge History Image)
    Input : frames (shape: [f, h, w])
    Output : mehi (shape: [f-1, h, w])
    """
    return [cv2.absdiff(frames[i], frames[i-1]) for i in range(1, len(frames))]

def construct_bow_features(sift_descriptors, kmeans):
    """
    Return Bag of Word features constructed from the given SIFT descriptors. 
    Input : descriptors (shape: [num_descriptors, 128])
    Output : bow (a numpy 1-d array (length: n_clusters of the given kmeans))
    """
    labels = kmeans.predict(sift_descriptors)
    bow_features = np.zeros((kmeans.n_clusters, 1), dtype=np.float32)
    bow_features = np.bincount(labels, minlength=kmeans.n_clusters).astype(np.float32).reshape(kmeans.n_clusters, 1)
    scaler = StandardScaler()
    warnings.filterwarnings("ignore")
    bow_features = scaler.fit_transform(bow_features)
    warnings.filterwarnings("default")
    return bow_features.flatten()


def auxiliary_feature(gray): 
    """
    Extract the auxiliary features from the gray images (gray: shape [N, 1, f, h, w])
    Output shape : [N, 30] 
    """
    N, _, f, h, w = gray.shape
    bow_features_list = []
    gray_descriptors_list = []
    mehi_descriptors_list = []
    kmeans1 = KMeans(n_clusters=20, random_state=0) # kmeans for gray image
    kmeans2 = KMeans(n_clusters=10, random_state=0) # kmeans for MEHI image
    sift = cv2.SIFT_create()
    
    for i in tqdm(range(N), desc="\taux) SIFT"):
        frames = [(gray[i, 0, j].cpu().numpy() * 255).astype(np.uint8) for j in range(f)]
        # compute SIFT of gray image & MEHI
        gray_descriptors_list.append(compute_sift_descriptors(frames, sift=sift))
        mehi = compute_mehi(frames)
        mehi_descriptors_list.append(compute_sift_descriptors(mehi, sift=sift))
    
    kmeans1.fit(np.concatenate(gray_descriptors_list, axis=0))
    kmeans2.fit(np.concatenate(mehi_descriptors_list, axis=0))

    for i in tqdm(range(N), desc="\taux) BoW"):
        # Construct BoW features
        bow_features_gray = list(construct_bow_features(gray_descriptors_list[i], kmeans1))
        bow_features_mehi = list(construct_bow_features(mehi_descriptors_list[i], kmeans2))
        combined_bow_features = torch.tensor(bow_features_gray + bow_features_mehi) #np.concatenate((bow_features_gray, bow_features_mehi), axis=0)
        bow_features_list.append(combined_bow_features)
    
    auxiliary_tensor = torch.stack(bow_features_list, dim=0).type(torch.float32)
    return auxiliary_tensor



#* 3D-CNN Model
class Original_Model(nn.Module):
    """
    3D-CNN model designed by the '3D-CNN for HAR' paper. 
    Input Shape: (N, C_in=1, Dimension=5f-2, Height=h, Width=w)
    """
    def __init__(self, verbose=False, mode='KTH'):
        """You need to give the dataset type as 'mode', which is one of 'KTH' or 'TRECVID'."""
        super(Original_Model, self).__init__()
        self.verbose = verbose
        self.mode = mode
        if self.mode == 'KTH':
            self.f = 9
        elif self.mode == 'TRECVID':
            self.f = 7
        else:
            raise ValueError("This mode is not available. Choose one of KTH or TRECVID.")

        self.dim = self.f * 5 - 2
        self.dim1, self.dim2 = (self.dim-10)*2, (self.dim-20)*6
        self.sift = cv2.SIFT_create()
        self.kmeans1 = KMeans(n_clusters=20, random_state=0) # kmeans for gray image
        self.kmeans2 = KMeans(n_clusters=10, random_state=0) # kmeans for gray image

        self.dropout = nn.Dropout(p=0.5)

        if self.mode == 'KTH':
            self.classes = 6
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,9,7), stride=1)
            self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,7), stride=1)
            self.pool1 = nn.MaxPool2d(3)
            self.pool2 = nn.MaxPool2d(3)
            self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=128, kernel_size=(6,4), stride=1)
            self.fc1 = nn.Linear(158, 30 + self.classes, bias=False)

        elif self.mode == 'TRECVID':
            self.classes = 3
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,7,7), stride=1)
            self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,6), stride=1)
            self.pool1 = nn.MaxPool2d(2)
            self.pool2 = nn.MaxPool2d(3)
            self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=128, kernel_size=(7,4), stride=1)
            self.fc1 = nn.Linear(158, 30 + self.classes, bias=False)

    def forward(self, x, aux):
        if self.verbose: print("연산 전:\t", x.size())
        assert x.size()[1] == 1

        (x1, x2, x3, x4, x5) = torch.split(x, [self.f,self.f,self.f,self.f-1,self.f-1], dim=2)
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

        (x1, x2, x3, x4, x5) = torch.split(x, [self.f-2,self.f-2,self.f-2,self.f-3,self.f-3], dim=2)
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

        x = self.dropout(x) # dropout (opt)
        x = x.view(-1, 128)
        aux_added = torch.cat([x,aux], dim=1)
        x = self.fc1(aux_added)[:,:self.classes]
        if self.verbose: print("fc1 연산 후:\t", x.shape)

        return x

    # def compute_sift_descriptors(self, frames):
    #     """Make sure to have more than 20 of output descriptors """
    #     all_descriptors_list = [np.random.rand(20, 128).astype(np.float32)]
    #     # for image in frames:
    #     #     keypoints, descriptors = sift.detectAndCompute(image, None)
    #     #     if descriptors is not None:
    #     #         all_descriptors_list.append(descriptors)
    #     all_descriptors_list += [self.sift.detectAndCompute(image, None)[1] for image in frames]
    #     all_descriptors = np.vstack([x for x in all_descriptors_list if x is not None])
    #     if len(all_descriptors) > 40 : 
    #         return all_descriptors[20:]
    #     else :
    #         return all_descriptors[len(all_descriptors)-20:]

    # def compute_mehi(self, frames):
    #     return [cv2.absdiff(frames[i], frames[i-1]) for i in range(1, len(frames))]
    
    # def construct_bow_features(self, kmeans, sift_descriptors):
    #     labels = kmeans.predict(sift_descriptors)
    #     bow_features = np.zeros((kmeans.n_clusters, 1), dtype=np.float32)
    #     for label in labels:
    #         bow_features[label, 0] += 1
    #     scaler = StandardScaler()
    #     warnings.filterwarnings("ignore")
    #     bow_features = scaler.fit_transform(bow_features)
    #     return bow_features.flatten()


    # def auxiliary_feature(self, x1): 
    #     """
    #     Extract the auxiliary features from the gray images (x1: shape [N, 1, f, h, w])
    #     Output shape : [N, 30] 
    #     """
    #     N, _, f, h, w = x1.shape
    #     bow_features_list = []
    #     gray_descriptors_list = [0 for i in range(N)]
    #     mehi_descriptors_list = [0 for i in range(N)]
        
    #     for i in range(N):
    #         frames = [(x1[i, 0, j].cpu().numpy() * 255).astype(np.uint8) for j in range(f)]
    #         # import time
    #         # start_time = time.time()
    #         # compute SIFT of gray image & MEHI
    #         gray_descriptors_list[i] = self.compute_sift_descriptors(frames)
    #         mehi = self.compute_mehi(frames)
    #         mehi_descriptors_list[i] = self.compute_sift_descriptors(mehi)
    #         # print('SIFT time : ', time.time()-start_time)
    #         # start_time = time.time()
        
    #     self.kmeans1.fit(np.concatenate(gray_descriptors_list, axis=0))
    #     self.kmeans2.fit(np.concatenate(mehi_descriptors_list, axis=0))
    #     # print('kmeans fit time : ', time.time()-start_time)
    #     # start_time = time.time()

    #     for i in range(N):
    #         # Construct BoW features
    #         bow_features_gray = self.construct_bow_features(self.kmeans1, gray_descriptors_list[i])
    #         bow_features_mehi = self.construct_bow_features(self.kmeans2, mehi_descriptors_list[i])
    #         combined_bow_features = np.concatenate((bow_features_gray, bow_features_mehi), axis=0)
    #         bow_features_list.append(combined_bow_features)
        
    #     auxiliary_tensor = torch.tensor(np.array(bow_features_list), dtype=torch.float32).to(x1.device)
    #     return auxiliary_tensor


