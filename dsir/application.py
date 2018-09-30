import glob
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage import exposure
from skimage.io import imread
from os.path import isfile, join
from torch.autograd import Variable
from skimage.transform import resize


###### CONSTANTS ######
# red square
red_x = 30
red_y = 2
red_sqr = 26
# green square
green_x = red_x + (20/12.5)
green_y = red_y
green_sqr = 15
model_path = join("Deep-Learning-Super-Resolution-Image-Reconstruction-DSIR",
                 "model", "autoencoder_model.pt")


# Useful functions, copied from the DSIR GitHub repo - as it lacks an importable module
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv4 = nn.Conv2d(8, 1, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(1)
        self.convt1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.convt2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.convt3 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                nn.init.normal(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.bn1(x)))  # out [16, 104, 104, 1]
        x = self.conv2(x)
        x = self.pool(F.relu(self.bn2(x))) # out [8, 52, 52, 1]
        x = self.conv3(x)
        x = self.pool(F.relu(self.bn2(x))) # out [8, 26, 26, 1]

        x = self.convt1(x)
        x = F.relu(self.bn2(x)) # out [8, 52, 52, 1]
        x = self.convt2(x)
        x = F.relu(self.bn1(x)) # out [16, 104, 104, 1]
        x = self.convt3(x)
        x = F.relu(self.bn2(x)) # out [8, 208, 208, 1]
        x = self.conv4(x)
        x = F.relu(self.bn4(x)) # out [1, 208, 208, 1]

        return x

###### Preprocess and decode image data ######
def decode_img(img_zoom):
    img_zoom_scale = resize(img_zoom, (red_sqr * 8, red_sqr * 8), preserve_range=True,
                            mode='constant', anti_aliasing=True)
    img_zoom_scale = (
                     img_zoom_scale - img_zoom_scale.mean()) / img_zoom_scale.std()  # Normalization
    img_zoom_scale = Variable(torch.FloatTensor(img_zoom_scale).view(1, 1, 208, 208))
    return img_zoom_scale

def img_rescale(img, p1=0, p2=99.9):
    p1_value, p2_value = np.percentile(img, (p1, p2))
    img_rescaled = exposure.rescale_intensity(img, in_range=(p1_value, p2_value))
    return img_rescaled


# Load in the model into memory
model = autoencoder()
model.load_state_dict(torch.load(model_path, map_location="cpu"))


# The actual functions to run the model
def run(all_images):
    """Apply the encoder to a set of images

    Args:
        all_images (string): Paths to all the images in a dataset
    Returns:
        (ndarray) Accumulated results from all images
        """

    # Encode all images and accumulate result
    image = np.zeros((208, 208))
    for frame in all_images:
        img = imread(frame)[red_y:(red_y+red_sqr),red_x:(red_x+red_sqr)]
        decoded_img = model(decode_img(img)).data.view(208, 208)
        image += decoded_img

    return image


def test_run():
    data_path = "./data/"
    files = glob.glob(join(data_path, '*.tif'))
    out = run(files)
    assert out.shape == (208, 208)

    return out


if __name__ == '__main__':
    out = test_run()
    print(out)
