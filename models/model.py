import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.decomposition import PCA

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


def compute_dense_sift(frames, sift=None, step_size=6, sizes=[7, 16]):
    """
    Extract SIFT descriptors from every 6 pixels of consecutive frames. 
    Input : frames (shape: [f, h, w])
    Output : descriptors, keypoints (descriptors shape: [num_descriptors, 128] / keypoints shape: [num_descriptors, 2]). 
        Each descriptor feature has 128 dimension and each keypoint has 2 dimension (x, y).
        The number of descriptors depends on the SIFT result, but guarantees to have more than 512. 
    """
    min_desc, dimension = 512, 128
    if sift is None :
        sift = cv2.SIFT_create()

    all_descriptors_list = []
    all_keypoints_list = []
    for size in sizes:
        keypoints = [[cv2.KeyPoint(x, y, size) for y in range(0, image.shape[0], step_size)
                                                for x in range(0, image.shape[1], step_size)] for image in frames]
        output = [sift.compute(image, keypoints[i]) for i, image in enumerate(frames)]
        all_descriptors_list += [np.concatenate([[desc for desc in output[f][1]] for f in range(len(frames))])]
        all_keypoints_list += [np.concatenate([[kp.pt for kp in output[f][0]] for f in range(len(frames))])]
    
    descriptors, keypoints = np.vstack(all_descriptors_list), np.vstack(all_keypoints_list)
    
    assert len(descriptors) == len(keypoints) and len(descriptors) >= min_desc
    assert descriptors.shape[1] == dimension

    return descriptors, keypoints

def compute_mehi(frames):
    """
    Compute MEHI (Motion Edge History Image)
    Input : frames (shape: [f, h, w])
    Output : mehi (shape: [f-1, h, w])
    """
    return [cv2.absdiff(frames[i], frames[i-1]) for i in range(1, len(frames))]

def softly_quantize_descriptors(descriptors, kmeans):
    """Function to softly quantize descriptors using a codebook"""
    distances = kmeans.transform(descriptors)
    soft_assignments = np.exp(-distances)
    soft_assignments /= soft_assignments.sum(axis=1, keepdims=True)
    return soft_assignments

def construct_spm_features(descriptors, keypoints, image_shape, kmeans, pyramid_levels=[(2, 2), (3, 4)], num_clusters=512):
    """
    Function to construct SPM features.
    output : spm_features (numpy array of shape : (8192,))
    """
    h, w = image_shape
    spm_features = []
    for level in pyramid_levels:
        num_cells_x, num_cells_y = level
        cell_h, cell_w = h // num_cells_y, w // num_cells_x
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                cell_descriptors = descriptors[(j * cell_w <= keypoints[:, 0]) & (keypoints[:, 0] < (j + 1) * cell_w) 
                                            & (i * cell_h <= keypoints[:, 1]) & (keypoints[:, 1] < (i + 1) * cell_h)]

                if cell_descriptors.size != 0:
                    soft_assignments = softly_quantize_descriptors(cell_descriptors, kmeans)
                    spm_feature = soft_assignments.sum(axis=0)
                    spm_features.append(spm_feature)
                else:
                    spm_features.append(np.zeros(num_clusters))
    spm_features = np.concatenate(spm_features)
    
    assert len(spm_features) == 16*512  # 8192 == 16 x 512
    return spm_features


def auxiliary_feature(gray, num_clusters=512): 
    """
    Extract the auxiliary features from the gray images (gray: shape [N, 1, f, h, w])
    Output shape : [N, 300] 
    """
    N, _, f, h, w = gray.shape
    all_spm_features_gray = []
    all_spm_features_mehi = []
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.random.rand(10000, 128)) # kmeans for gray image
    pca = PCA(n_components=150).fit(np.random.rand(1000, 8192))
    sift = cv2.SIFT_create()
    
    for i in tqdm(range(N), desc="auxiliary", bar_format='\t{desc:<12}:{percentage:3.0f}%|{bar:60}{r_bar}'):
        frames = [(gray[i, 0, j].cpu().numpy() * 255).astype(np.uint8) for j in range(f)]

        # compute SIFT from gray image, MEHI
        dense_sift_descriptors_gray, keypoints_gray = compute_dense_sift(frames, sift=sift) # shape: (num_desc, 128), (num_desc, 2)
        mehi = compute_mehi(frames)
        dense_sift_descriptors_mehi, keypoints_mehi = compute_dense_sift(mehi, sift=sift)   # shape: (num_desc, 128), (num_desc, 2)

        # Construct SPM features from the SIFT descriptors
        spm_features_gray = construct_spm_features(dense_sift_descriptors_gray, keypoints_gray, (h, w), kmeans, num_clusters=num_clusters) # shape: (16, 512)
        spm_features_mehi = construct_spm_features(dense_sift_descriptors_mehi, keypoints_mehi, (h, w), kmeans, num_clusters=num_clusters)
        all_spm_features_gray.append(spm_features_gray)
        all_spm_features_mehi.append(spm_features_mehi)

    # Concatenate gray and MEHI features, and apply PCA
    spm_features_gray = np.vstack(all_spm_features_gray) if all_spm_features_gray else np.zeros((N, 8192)) # shape: (N, 8192)
    spm_features_mehi = np.vstack(all_spm_features_mehi) if all_spm_features_mehi else np.zeros((N, 8192)) # shape: (N, 8192)
    spm_features_gray_pca = torch.tensor(pca.transform(spm_features_gray)) # shape: (N, 150)
    spm_features_mehi_pca = torch.tensor(pca.transform(spm_features_mehi)) # shape: (N, 150)
    auxiliary_tensor = torch.concat([spm_features_gray_pca, spm_features_mehi_pca], dim=1).type(torch.float32) # shape: (N, 300)

    return auxiliary_tensor



#* 3D-CNN Model
class Original_Model(nn.Module):
    """
    3D-CNN model designed by the '3D-CNN for HAR' paper. 
    Input Shape: (N, C_in=1, Dimension=5f-2, Height=h, Width=w)
    """
    def __init__(self, verbose=False, mode='KTH', add_reg=True):
        """You need to give the dataset type as 'mode', which is one of 'KTH' or 'TRECVID'."""
        super(Original_Model, self).__init__()
        self.verbose = verbose
        self.mode = mode
        self.add_reg = add_reg
        if self.mode == 'KTH':
            self.f = 9
        elif self.mode == 'TRECVID':
            self.f = 7
        else:
            raise ValueError("This mode is not available. Choose one of KTH or TRECVID.")

        self.dim = self.f * 5 - 2
        self.dim1, self.dim2 = (self.dim-10)*2, (self.dim-20)*6
        self.last_dim, self.aux_dim = 128, 300

        self.dropout = nn.Dropout(p=0.5)

        if self.mode == 'KTH':
            self.classes = 6
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,9,7), stride=1)
            self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,7), stride=1)
            self.pool1 = nn.MaxPool2d(3)
            self.pool2 = nn.MaxPool2d(3)
            self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=self.last_dim, kernel_size=(6,4), stride=1)
            self.fc1 = nn.Linear(self.last_dim + self.aux_dim if self.add_reg else self.last_dim, self.classes, bias=False)

        elif self.mode == 'TRECVID':
            self.classes = 3
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,7,7), stride=1)
            self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,6), stride=1)
            self.pool1 = nn.MaxPool2d(2)
            self.pool2 = nn.MaxPool2d(3)
            self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=self.last_dim, kernel_size=(7,4), stride=1)
            self.fc1 = nn.Linear(self.last_dim + self.aux_dim if self.add_reg else self.last_dim, self.classes, bias=False)

    def forward(self, x, aux):
        if self.verbose: print("연산 전:\t", x.size())
        assert x.size()[1] == 1

        (x1, x2, x3, x4, x5) = torch.split(x, [self.f,self.f,self.f,self.f-1,self.f-1], dim=2)
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        x3 = self.conv1(x3)
        x4 = self.conv1(x4)
        x5 = self.conv1(x5)
        # x1 = F.relu(self.conv1(x1))
        # x2 = F.relu(self.conv1(x2))
        # x3 = F.relu(self.conv1(x3))
        # x4 = F.relu(self.conv1(x4))
        # x5 = F.relu(self.conv1(x5))
        x = torch.cat([x1, x2, x3, x4, x5], dim=2)
        if self.verbose: print("conv1 연산 후:\t", x.shape)

        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.pool1(F.relu(x))
        x = x.view(-1, 2, self.dim1//2, x.shape[2], x.shape[3])
        if self.verbose: print("pool1 연산 후:\t", x.shape)

        (x1, x2, x3, x4, x5) = torch.split(x, [self.f-2,self.f-2,self.f-2,self.f-3,self.f-3], dim=2)
        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)
        x5 = self.conv2(x5)
        x = torch.cat([x1, x2, x3, x4, x5], dim=2)
        if self.verbose: print("conv2 연산 후:\t",x.shape)

        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.pool2(F.relu(x))
        x = x.view(-1, self.dim2, x.shape[2], x.shape[3])
        if self.verbose: print("pool2 연산 후:\t", x.shape)

        x = F.relu(self.conv3(x))
        if self.verbose: print("conv3 연산 후:\t", x.shape)

        if self.add_reg:
            x = torch.cat((x.view(-1, self.last_dim),aux), dim=1)
        else:
            x = x.view(-1, self.last_dim)
        x = self.fc1(x)# [:,:self.classes]
        if self.verbose: print("fc1 연산 후:\t", x.shape)

        return x
