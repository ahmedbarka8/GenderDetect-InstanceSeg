import os
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
import cv2
from detectron2.utils.visualizer import Visualizer


dataset_dir = r"C:\Users\Ahmed\LV-MHP-v2\train"

register_coco_instances(
    "dataset",
    {},  # optional metadata dict
    os.path.join(dataset_dir, "coco_cvat.json"),
    os.path.join(r"C:\Users\Ahmed\LV-MHP-v2\train\images")
)


my_dataset_metadata = MetadataCatalog.get("dataset")
dataset_dicts = DatasetCatalog.get("dataset")

print("Number of images:", len(dataset_dicts))
for d in dataset_dicts:
    print("File:", d["file_name"])
    for ann in d["annotations"]:
        print("  Bbox:", ann["bbox"], "Category:", ann["category_id"])



for d in dataset_dicts:
    img = cv2.imread(d["file_name"])
    if img is None:
        continue
    v = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata, scale=0.5)
    out = v.draw_dataset_dict(d)
    cv2.imshow("Preview", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

