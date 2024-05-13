import torch
from torchsummary import summary
import numpy as np
import cv2
import os
from Models import Original_Model, hardwire_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

def main():

    # Example Youtube Video Download
    filename='./example_video/video_example.mp4'
    Download(filename=filename)
    
    # Extract the input frames from the raw video data
    # N, f, h, w = (10, 7, 60, 40) # TRECVID shape
    N, f, h, w = (10, 9, 80, 60) # KTH shape
    input_frames = np.zeros((N, f, h, w, 3), dtype='float32') 
    cap = cv2.VideoCapture(filename)
    for i in range(N):
        for j in range(f):
            _, frame = cap.read()
            input_frames[i,j,:,:,:] = frame[100:100+h, 200:200+w, :]
    
    # Hardwired layer
    inputs = hardwire_layer(input_frames).to(device)  
    cnn = Original_Model(verbose=True, input_dim=f).to(device)
    summary(cnn, (1,5*f-2,h,w))  # Input Size: (N, C_in=1, Dimension=5*f-2, Height=h, Width=w)


def Download(link="https://youtu.be/GUPu3ilbfbE?feature=shared", filename='./example_video/video_example.mp4'):
    """Download the example youtube video. """
    from pytube import YouTube

    if os.path.exists(filename):
        print("Already downloaded:", filename)
        return
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_lowest_resolution()
    try:
        youtubeObject.download(filename=filename)
    except:
        print("An error has occurred")
    print("Download is completed successfully")



if __name__ == '__main__':
    main()
