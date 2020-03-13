#!/usr/bin/env python3

"""
Made by @nizhib
"""

import base64
import io
import logging
import sys
sys.path.append("models/gca")
import time
from http import HTTPStatus

import numpy as np
import cv2
import requests
from flask import Flask, request, jsonify
from imageio import imsave
from PIL import Image
from waitress import serve

from api import Segmentator
from models.utils import gen_trimap
from gca import GCA
LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = '[%(asctime)s] %(name)s:%(lineno)d: %(message)s'

logging.basicConfig(format=LOGGING_FORMAT, level=LOGGING_LEVEL)

segmentator = Segmentator()
segmentator.load('resource/unet_resnext50.pth')
image_matter = GCA()
SEGMENTATION_SIZE = (800, 564)
MATTING_SIZE = (240, 320)

app = Flask(__name__)
logger = logging.getLogger(__file__)

def serialize(image):
    fmem = io.BytesIO()
    imsave(fmem, image, 'png')
    fmem.seek(0)
    image64 = base64.b64encode(fmem.read()).decode('utf-8')
    return image64

def deserialize(data, is_url = True):
    if is_url:
        blob = io.BytesIO(requests.get(data).content)
    else:
        blob = io.BytesIO(requests.get(data))
    img = Image.open(blob).convert('RGB')
    return img

@app.route('/segment', methods=['POST'])
def handle():
    start = time.time()
    status = HTTPStatus.OK
    result = {'success': False}

    try:
        data = request.json
        if 'image' in data:
            img = deserialize(data['image'], False)
        elif 'url' in data:
            img = deserialize(data['url'])
        else:
            raise ValueError("No image source found in request fields:")

        if "bg_url" in data:
            bg_img = deserialize(data['bg_url'])
            bg_img = cv2.resize(np.array(bg_img).astype(np.uint8), MATTING_SIZE)

        # prepare data
        mask = segmentator.predict(img)
        mask[mask >= 0.1] = 1
        mask[mask < 0.1] = 0
        mask = (mask * 255).astype(np.uint8)
        # inference
        trimap = gen_trimap(mask)
        alpha = image_matter.inference(cv2.resize(np.array(img).astype(np.uint8), SEGMENTATION_SIZE), cv2.resize(trimap, SEGMENTATION_SIZE))
        range_alpha = cv2.resize(alpha / 255, MATTING_SIZE)
        merged_img = range_alpha[..., None] * cv2.resize(np.array(img).astype(np.uint8), MATTING_SIZE) + (1 - range_alpha)[..., None] * bg_img

        trimap64 = serialize(trimap)
        alpha64 = serialize(cv2.resize(alpha, MATTING_SIZE))
        merged_img = serialize(merged_img)

        result['data'] = {'trimap': trimap64, "alpha": alpha64, "merged": merged_img}
        result['success'] = True
    except Exception as e:
        logger.exception(e)
        result['message'] = str(e)
        status = HTTPStatus.INTERNAL_SERVER_ERROR

    result['total'] = time.time() - start

    return jsonify(result), status


if __name__ == '__main__':
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 5000

    serve(app, host='0.0.0.0', port=port)
