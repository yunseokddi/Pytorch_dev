import numpy as np
import torch



def Normalize(image):
    image_copy = np.copy(image)

    image_copy = image_copy / 255.0

    return image_copy


def ToTensor(image):
    if (len(image.shape) == 2):
        # add that third color dim
        image = image.reshape(image.shape[0], image.shape[1], 1)

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))

    return torch.from_numpy(image)