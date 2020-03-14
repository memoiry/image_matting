import torch
import cv2
import numpy as np

def size(net):
    pp = 0
    for p in list(net.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def gen_trimap(segmentation_mask, k_size = 7, iterations = 6):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(segmentation_mask, kernel, iterations=iterations)
    eroded = cv2.erode(segmentation_mask, kernel, iterations=iterations)
    trimap = np.zeros(segmentation_mask.shape, dtype=np.uint8)
    trimap.fill(128)

    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
    return trimap

def inspect_model(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images = torch.randn(4, 3, 64, 64).to(device)

    net = model(num_classes=2).to(device)

    logits = net(images)
    if images.shape[-2:] != logits.shape[-2:]:
        raise ValueError('Output sized {} while {} expected'.format(logits.shape[-2:], images.shape[-2:]))

    print(size(net), model.__name__, sep='\t')
