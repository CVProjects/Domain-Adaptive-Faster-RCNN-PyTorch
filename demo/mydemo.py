import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


from io import BytesIO
from PIL import Image
import numpy as np

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "/home/imlab/haeyeon/cvproject/Domain-Adaptive-Faster-RCNN-PyTorch/configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE","cpu","MODEL.WEIGHT","/home/imlab/haeyeon/cvproject/Domain-Adaptive-Faster-RCNN-PyTorch/output/citytosnowy/model_final.pth"])


coco_demo = COCODemo(cfg,
min_image_size = 800,
confidence_threshold=0.5,
)
def load(imagepath):
    pil_image = Image.open(imagepath)
    image = np.array(pil_image)[:,:,[2,1,0]]
    return image

def imsave(img):
    plt.imsave("detected_snowy.jpg",img[:,:,[2,1,0]])
    

image_path = "/home/imlab/haeyeon/cvproject/foggycityscape/images/leftImg8bit/test/munich/munich_000395_000019_leftImg8bit_foggy_beta_0.02.png"
image = load(image_path)

predictions = coco_demo.run_on_opencv_image(image)
imsave(predictions)

