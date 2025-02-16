import os,cv2, random
from detectron2.data.datasets import register_coco_instances
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader
)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor,DefaultTrainer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode


if __name__ == "__main__":
    # 1) Register your train set (COCO format) once:
    train_dir = r"C:\Users\Ahmed\LV-MHP-v2\train"
    train_json = os.path.join(train_dir, "coco_cvat.json")
    register_coco_instances(
        "mydata_train",
        {},
        train_json,
        train_dir+"\\images"
    )
    # Then set the categories:
    MetadataCatalog.get("mydata_train").thing_classes = ['man', 'woman', 'child', 'person']

    # 2) Register your val set (COCO format) once:
    val_dir = r"C:\Users\Ahmed\LV-MHP-v2\val"
    val_json = os.path.join(val_dir, "coco_cvat.json")
    register_coco_instances(
        "mydata_val",
        {},
        val_json,
        val_dir+"\\images"
    )
    MetadataCatalog.get("mydata_val").thing_classes = ['man', 'woman', 'child', 'person']

    # 3) Now configure
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    )
    # Set the train & val dataset names
    cfg.DATASETS.TRAIN = ("mydata_train",)
    cfg.DATASETS.TEST = ("mydata_val",)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    )
    cfg.MODEL.WEIGHTS = r"output/Gender/model_final.pth"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 2000  # train longer if possible
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.OUTPUT_DIR = r"./output/Gender"

    # # 4) Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # 5) Evaluation
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("mydata_val", output_dir="./output/gender_eval")
    val_loader = build_detection_test_loader(cfg, "mydata_val")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print("Eval results:", results)

    # 6) Visualize some predictions on the val set:
    val_dicts = DatasetCatalog.get("mydata_val")
    val_meta = MetadataCatalog.get("mydata_val")

    for d in random.sample(val_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get("mydata_train"),
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("window",out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
