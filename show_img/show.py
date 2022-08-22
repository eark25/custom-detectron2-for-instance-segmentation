# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode

def get_crack_dicts(img_dir):
    json_file = '/root/detectron2/show_img/example1.json'
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    # loop through each image

    record = {}
    
    filename = os.path.join('/root/detectron2/show_img/', imgs_anns['images'][0]['file_name'])
    height, width = cv2.imread(filename).shape[:2]
    
    # common fields
    record["file_name"] = filename
    record["image_id"] = imgs_anns['images'][0]['id']
    record["height"] = height
    record["width"] = width

    for obj in imgs_anns['annotations']:
        px = []
        py = []
        obj['bbox_mode'] = BoxMode.XYXY_ABS
        for mask in obj['segmentation']:
            for idx in range(len(mask)):
                if (idx % 2) == 0:
                    px.append(mask[idx])
                else:
                    py.append(mask[idx])
            # print('mask', mask)
            # print('px', px)
            # print('py', py)
        obj['bbox'] = [np.min(px), np.min(py), np.max(px), np.max(py)]
    
    record["annotations"] = imgs_anns['annotations']
    dataset_dicts.append(record)
    # return list[dict]
    return dataset_dicts

for d in ["train"]:
    DatasetCatalog.register("crack_" + d, lambda d=d: get_crack_dicts("balloon/" + d))
    MetadataCatalog.get("crack_" + d).set(thing_classes=["outlier", "crack"], evaluator_type="coco")
crack_metadata = MetadataCatalog.get("crack_train")
crack_train_dataset = len(DatasetCatalog.get("crack_train"))
# evaluator_type = MetadataCatalog.get("balloon_val").evaluator_type
# print(balloon_metadata)
# print(len(balloon_train_dataset))
# import sys
# sys.exit(0)

# To verify the dataset is in correct format, let's visualize the annotations of randomly selected samples in the training set:
dataset_dicts = get_crack_dicts("balloon/train")
# print(dataset_dicts)
# import sys
# sys.exit(0)
for d in random.sample(dataset_dicts, 1):
    # print(d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=crack_metadata, scale=1.0)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite('example1.jpg', out.get_image()[:, :, ::-1])