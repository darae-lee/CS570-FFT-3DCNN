import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def hardwire_layer(input, device, verbose=False):
    """
    Preprocess the given consecutive input (grayscaled) frames into 5 different styles. 
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

    # Clone the input tensor and move it to the specified device
    input_cloned = input.clone().to(device)
    
    # Normalize each type of data separately
    gray_mean = input_cloned.mean()
    gray_std = input_cloned.std()
    gray_normalized = (input_cloned - gray_mean) / gray_std

    x_gradient = torch.gradient(input_cloned, dim=[2,3])[0]
    x_gradient_mean = x_gradient.mean()
    x_gradient_std = x_gradient.std()
    x_gradient_normalized = (x_gradient - x_gradient_mean) / x_gradient_std

    y_gradient = torch.gradient(input_cloned, dim=[2,3])[1]
    y_gradient_mean = y_gradient.mean()
    y_gradient_std = y_gradient.std()
    y_gradient_normalized = (y_gradient - y_gradient_mean) / y_gradient_std

    x_flow = []
    y_flow = []
    for i in range(N):
        prevs = input_cloned[i, :-1].detach().cpu().numpy()
        nexts = input_cloned[i, 1:].detach().cpu().numpy()
        for prev_, next_ in zip(prevs, nexts):
            flow = cv2.calcOpticalFlowFarneback(prev_, next_,
                                                None, 0.5, 3, 15, 3, 5, 1.1,
                                                cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            x_flow.append(torch.tensor(flow[...,0]))
            y_flow.append(torch.tensor(flow[...,1]))
    x_flow = torch.stack(x_flow, dim=0).reshape(N, f-1, h, w).to(device)
    y_flow = torch.stack(y_flow, dim=0).reshape(N, f-1, h, w).to(device)

    x_flow_mean = x_flow.mean()
    x_flow_std = x_flow.std()
    x_flow_normalized = (x_flow - x_flow_mean) / x_flow_std

    y_flow_mean = y_flow.mean()
    y_flow_std = y_flow.std()
    y_flow_normalized = (y_flow - y_flow_mean) / y_flow_std
    
    # Concatenate all the preprocessed data
    hardwired[:, :f, :, :] = gray_normalized
    hardwired[:, f:f*2, :, :] = x_gradient_normalized
    hardwired[:, f*2:f*3, :, :] = y_gradient_normalized
    hardwired[:, f*3:f*4-1, :, :] = x_flow_normalized
    hardwired[:, f*4-1:, :, :] = y_flow_normalized

    hardwired = hardwired.unsqueeze(dim=1)
    if verbose: print("After hardwired layer :\t", hardwired.shape)
    return hardwired


def compute_dense_sift(inputs, sift=None, pbar=None):
    """
    Extract SIFT descriptors from every 6 pixels of consecutive frames. 
    Input : inputs (shape: [N, f, h, w]) 
    Output : descriptors, keypoints (descriptors shape: [N, num_descriptors, 128] / keypoints shape: [num_descriptors, 2]). 
        Each descriptor feature has 128 dimension and each keypoint has 2 dimension (x, y).
        The number of descriptors depends on the SIFT result, but guarantees to have more than 512. 
    """
    N, f, h, w = inputs.shape
    step_size, sizes= 6, [7, 16]
    min_desc, dimension = 512, 128
    if sift is None :
        sift = cv2.SIFT_create()

    all_descriptors_list = []
    all_keypoints_list = []
    for size in sizes:
        keypoints = [cv2.KeyPoint(x, y, size) for y in range(0, h, step_size) for x in range(0, w, step_size)]
        if pbar: pbar.update(1)
        all_descriptors_list.append([[sift.compute(image, keypoints)[1] for image in frames] for frames in inputs])
        if pbar: pbar.update(10)
        all_keypoints_list.append(np.concatenate([[kp.pt for kp in keypoints] for _ in range(f)]))
        if pbar: pbar.update(1)
    descriptors, keypoints = np.reshape(all_descriptors_list, newshape=(N, -1, dimension)), np.concatenate(all_keypoints_list)
    
    assert descriptors.shape[1] == keypoints.shape[0] 
    assert descriptors.shape[1] >= min_desc
    assert descriptors.shape[2] == dimension and keypoints.shape[1] == 2

    return descriptors, keypoints

def compute_mehi(inputs):
    """
    Compute MEHI (Motion Edge History Image)
    Input : inputs (shape: [N, f, h, w])
    Output : mehi (shape: [N, f-1, h, w])
    """
    return np.abs(np.diff(inputs, axis=1))


def construct_spm_features(descriptors, keypoints, image_shape, kmeans, pbar=None):
    """
    Function to construct SPM features.
    inputs : descriptors: (N, num_features, 128),   keypoints: (num_features, 2)
    output : spm_features (numpy array of shape : (N, 8192))
    """
    N, _, _ = descriptors.shape
    h, w = image_shape
    pyramid_levels = [(2, 2), (3, 4)]
    spm_features = []
    
    # softly quantize descriptors
    distances = torch.tensor(np.stack([kmeans.transform(descriptors[n, :]) for n in range(N)])).to('cuda')
    
    soft_list = []
    for distance in distances:
        soft_score = torch.softmax(distance, dim=1)
        soft_list.append(soft_score.detach().cpu())
    soft_assignments = torch.stack(soft_list, dim=0)
    soft_assignments = soft_assignments.numpy()
    
    for level in pyramid_levels:
        num_cells_x, num_cells_y = level
        cell_h, cell_w = h // num_cells_y, w // num_cells_x
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                spm_feature = soft_assignments[:, (j * cell_w <= keypoints[:, 0]) & (keypoints[:, 0] < (j + 1) * cell_w) 
                                            & (i * cell_h <= keypoints[:, 1]) & (keypoints[:, 1] < (i + 1) * cell_h)].sum(axis=1)
                spm_features += [spm_feature]
                if pbar: pbar.update(1)
    spm_features = np.concatenate(spm_features, axis=1)
    
    assert spm_features.shape == (N, 16*512)  # 8192 == 16 x 512
    return spm_features


def auxiliary_feature(gray, num_clusters=512, verbose=False): 
    """
    Extract the auxiliary features from the gray images (gray: shape [N, 1, f, h, w])
    Output shape : [N, 300] 
    """
    N, _, f, h, w = gray.shape
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.random.rand(10000, 128)) # kmeans for gray image
    pca = PCA(n_components=150).fit(np.random.rand(1000, 8192))
    sift = cv2.SIFT_create()

    gray = (np.squeeze(gray).cpu().numpy() * 255).astype(np.uint8) # numpy array of shape (N, f, h, w)
    mehi = compute_mehi(gray)   # (N, f-1, h, w)

    # compute SIFT from gray image, MEHI
    pbar = tqdm(total=82, desc="Auxiliary") if verbose else None
    dense_sift_descriptors_gray, keypoints_gray = compute_dense_sift(gray, sift=sift, pbar=pbar) # shape: (N, num_desc, 128), (num_desc, 2)
    dense_sift_descriptors_mehi, keypoints_mehi = compute_dense_sift(mehi, sift=sift, pbar=pbar)   # shape: (N, num_desc, 128), (num_desc, 2)

    # Construct SPM features from the SIFT descriptors
    spm_features_gray = construct_spm_features(dense_sift_descriptors_gray, keypoints_gray, (h, w), kmeans, pbar=pbar) # shape: (N, 8192)
    spm_features_mehi = construct_spm_features(dense_sift_descriptors_mehi, keypoints_mehi, (h, w), kmeans, pbar=pbar)

    # Concatenate gray and MEHI features, and apply PCA
    spm_features_gray_pca = torch.tensor(pca.transform(spm_features_gray)) # shape: (N, 150)
    pbar.update(1)
    spm_features_mehi_pca = torch.tensor(pca.transform(spm_features_mehi)) # shape: (N, 150)
    pbar.update(1)

    auxiliary_tensor = torch.concat([spm_features_gray_pca, spm_features_mehi_pca], dim=1).type(torch.float32) # shape: (N, 300)
    pbar.close()

    assert auxiliary_tensor.shape == (N, 300)
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
            self.fc1 = nn.Linear(self.last_dim, self.aux_dim + self.classes if self.add_reg else self.classes, bias=False)

        elif self.mode == 'TRECVID':
            self.classes = 3
            self.conv1 = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(3,7,7), stride=1)
            self.conv2 = nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3,7,6), stride=1)
            self.pool1 = nn.MaxPool2d(2)
            self.pool2 = nn.MaxPool2d(3)
            self.conv3 = nn.Conv2d(in_channels=self.dim2, out_channels=self.last_dim, kernel_size=(7,4), stride=1)
            self.fc1 = nn.Linear(self.last_dim, self.aux_dim + self.classes if self.add_reg else self.classes, bias=False)

    def forward(self, x):
        if self.verbose: print("연산 전:\t", x.size())
        assert x.size()[1] == 1

        (x1, x2, x3, x4, x5) = torch.split(x, [self.f,self.f,self.f,self.f-1,self.f-1], dim=2)
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        x3 = self.conv1(x3)
        x4 = self.conv1(x4)
        x5 = self.conv1(x5)
        x = torch.cat([x1, x2, x3, x4, x5], dim=2)
        x = self.dropout(x)
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
        x = self.dropout(x)
        if self.verbose: print("conv2 연산 후:\t",x.shape)

        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.pool2(F.relu(x))
        x = x.view(-1, self.dim2, x.shape[2], x.shape[3])
        if self.verbose: print("pool2 연산 후:\t", x.shape)

        x = F.relu(self.conv3(x))
        if self.verbose: print("conv3 연산 후:\t", x.shape)

        # if self.add_reg:
        #     x = torch.cat((x.view(-1, self.last_dim)), dim=1)
        # else:
        x = self.dropout(x)
        x = x.view(-1, self.last_dim)
        x = self.fc1(x)# [:,:self.classes]
        if self.verbose: print("fc1 연산 후:\t", x.shape)

        return x
