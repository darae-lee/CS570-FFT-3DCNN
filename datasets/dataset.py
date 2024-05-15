import os
import pickle
import torch
from torch.utils.data import Dataset
import random
# from models.model import hardwire_layer
from tqdm import tqdm
from datasets.KTH import make_raw_dataset

CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5
}
    
class KTHDataset(Dataset):
    """
    return hardwired train dataset, test dataset.
    if random=True : randomly select 16 subjects for train dataset, and put remaining 9 subjects for test dataset.
    * No valid dataset. (Following the paper's dataset preprocessing method.)
    """
    def __init__(self, directory="kth-data", type="train", transform= None, frames = 9, seed=2, device=torch.device('cuda')) :
        self.directory = os.path.join(os.getcwd(), "datasets", directory)
        self.type = type
        self.device = device
        self.num_subjects = 25 # number of subjects
        
        if not os.path.exists(self.directory) or len(os.listdir(self.directory)) < self.num_subjects:
            print("Making dataset")
            make_raw_dataset(directory=directory, transform=transform, f=frames, device=self.device)
        assert len(os.listdir(self.directory)) == self.num_subjects
        
        random.seed(seed)
        self.subjects = random.sample(range(self.num_subjects), 16) # list of randomly sampled 16 training subjects
        if self.type == "test":
            self.subjects = list(set(range(self.num_subjects)) - set(self.subjects)) # list of the remaining 9 test subjects
        print(self.type, "dataset subjects:", self.subjects)
        
        self.dataset, self.labels = self.read_dataset() 
        
        
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


    def read_dataset(self):
        inputs = [] # Tensor shape: (N,f,c,h,w)
        labels = []
        
        for subject_id in tqdm(self.subjects, desc="reading data"):
            filepath = os.path.join(self.directory, str(subject_id)+".p")
            subject = pickle.load(open(filepath, "rb"))
            inputs += subject["input"]
            labels += subject["category"]
            
        inputs = torch.stack(inputs, dim=0)
        labels = torch.LongTensor([CATEGORY_INDEX[l] for l in labels])
    
        return inputs, labels
