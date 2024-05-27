from tqdm import tqdm
import os
import pickle
import re
from torchvision import transforms, io
import torch
import sys
import fire

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(root_dir, 'models'))

from model import hardwire_layer, auxiliary_feature


CATEGORIES = [
    "boxing",
    "handclapping",
    "handwaving",
    "jogging",
    "running",
    "walking"
]

base_transform = transforms.Compose([
    transforms.Resize((80, 60)),
    transforms.Grayscale(num_output_channels=1)
])

def make_raw_dataset(directory="kth-data-aux", transform=None, f=9, device=None):
    """
    Make a raw dataset(format: {subject_id}.p) into the given directory from the raw KTH video dataset.
    Dataset are divided according to the instruction at:
    http://www.nada.kth.se/cvap/actions/00sequences.txt
    """
    frames_idx = parse_sequence_file()
    if not transform :
        transform = base_transform

    subjects = 25
    raw_path = os.path.join(os.getcwd(), "datasets", "kth-human-motion")  # directory path that the KTH dataset videos are stored
    dir_path = os.path.join(os.getcwd(), "datasets", directory) # directory path that the processed dataset will be stored
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    print("Processing ...")
    filenames = [[] for _ in range(subjects)] 
    for category in CATEGORIES:
        # Get all files in current category's folder.
        folder_path = os.path.join(raw_path, category)
        cat_files = sorted(os.listdir(folder_path))
        subject_ids = [int(file.split("_")[0][6:])-1 for file in cat_files]
        for (file,s) in zip(cat_files, subject_ids):
            filenames[s].append(file) 
            
    for subject_id in range(11, subjects):
        categories = []
        input = []
        
        for filename in tqdm(filenames[subject_id], desc='subject ' + str(subject_id)):
            # Get category in this video.
            category = filename.split("_")[1]
            file_path = os.path.join(raw_path, category, filename)

            frames = io.read_video(file_path, output_format='TCHW')[0] / 255.0
            frames = transform(frames).split(1, dim=0)

            for seg in frames_idx[filename]:
                seg_frames = frames[seg[0]:seg[1]+1]
                seg_frames = torch.cat(seg_frames, dim=0)

                N, throw = seg_frames.shape[0] // f, seg_frames.shape[0] % f
                if throw > 0:
                    seg_frames = seg_frames[:-throw]
                seg_frames = seg_frames.reshape(N, f, seg_frames.shape[-2], seg_frames.shape[-1])
            
                categories.extend([category for _ in range(N)])
                input.append(seg_frames)

        input = torch.cat(input, dim=0)
            
        # hardwiring layer
        input = hardwire_layer(input, device, verbose=True).cpu() # Tensor shape : (N, f, h, w) -> (N, 1, 5f-2, h, w)
        gray_img = input[:,:,:f, :, :]
        input_aux = auxiliary_feature(gray_img)

        # save the data per each subject
        person_path = os.path.join(dir_path, str(subject_id))
        
        data = {
            "category": categories,
            "input": input, # Tensor shape : (N, 1, 5f-2, h, w)
            "subject": subject_id,
            "aux": input_aux, # Tensor shape : (N, 30)
        }
        pickle.dump(data, open("%s.p" % person_path, "wb"))
        
    

def parse_sequence_file():
    filepath = os.path.join(os.getcwd(), 'datasets/kth-human-motion/00sequences.txt')
    print("Parsing ", filepath)

    # Read 00sequences.txt file.
    with open(filepath, 'r') as content_file:
        content = content_file.read()

    # Replace tab and newline character with space, then split file's content
    # into strings.
    content = re.sub("[\t\n]", " ", content).split()

    # Dictionary to keep ranges of frames with humans.
    # Example:
    # video "person01_boxing_d1": [(1, 95), (96, 185), (186, 245), (246, 360)].
    frames_idx = {}

    # Current video that we are parsing.
    current_filename = ""

    for s in content:
        if s == "frames":
            # Ignore this token.
            continue
        elif s.find("-") >= 0:
            # This is the token we are looking for. e.g. 1-95.
            if s[len(s) - 1] == ',':
                # Remove comma.
                s = s[:-1]

            # Split into 2 numbers => [1, 95]
            idx = s.split("-")

            # Add to dictionary.
            if not current_filename in frames_idx:
                frames_idx[current_filename] = []
            frames_idx[current_filename].append((int(idx[0]), int(idx[1])))
        else:
            # Parse next file.
            current_filename = s + "_uncomp.avi"

    return frames_idx

if __name__ == "__main__":
    #TODO : add kth_download
    print("Making dataset")
    fire.Fire(make_raw_dataset())