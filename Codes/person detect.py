from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

class_ids = {
    "person": 0,
    "man":1,
    "woman":2,
    "child":3
}

def get_LVMHPV2_dicts(img_dir):
    json_file = os.path.join(img_dir, "data_list.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    reduced_img_dir = "\\".join(img_dir.split("\\")[:-1])+"\\"
    dataset_dicts = []
    print(len(imgs_anns))
    for idx, v in enumerate(imgs_anns):
        record = {}

        try:
            filename = os.path.join(reduced_img_dir, v['filepath'].split(
                '/home/notebook/code/personal/S9043252/Parsing-R-CNN/data/LV-MHP-v2/')[1]).replace('/',"\\")
            height, width = cv2.imread(filename).shape[:2]
        except:
            continue

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for box in v["bboxes"]:

            obj = {
                "bbox": [box["x1"], box["y1"], box["x2"], box["y2"]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_ids[box["class"]],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

if __name__ == "__main__":
    # dataset_dicts = get_LVMHPV2_dicts("C:\\Users\\Ahmed\\LV-MHP-v2\\val")
    # print(len(dataset_dicts))
    # dataset_dicts = get_LVMHPV2_dicts("C:\\Users\\Ahmed\\LV-MHP-v2\\train")
    for d in ["train", "val"]:
        DatasetCatalog.register("LVMHPV2_" + d, lambda d=d: get_LVMHPV2_dicts("C:\\Users\\Ahmed\\LV-MHP-v2\\" + d))
        MetadataCatalog.get("LVMHPV2_" + d).set(thing_classes=["person","man","woman","child"])
    LVMHP_meta = MetadataCatalog.get("LVMHPV2_train")

    cfg = get_cfg()
    cfg.MODEL.WEIGHTS = r"output/person_Det/model_final.pth"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("LVMHPV2_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.OUTPUT_DIR = "./output/person_Det"
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_LVMHPV2_dicts("C:\\Users\\Ahmed\\LV-MHP-v2\\val")
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=LVMHP_meta,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("window",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    evaluator = COCOEvaluator("LVMHPV2_val", output_dir="output/Person_Det_val")
    val_loader = build_detection_test_loader(cfg, "LVMHPV2_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))