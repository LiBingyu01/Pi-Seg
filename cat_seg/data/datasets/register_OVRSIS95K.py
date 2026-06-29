import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

OVRSIS95K_CATEGORIES = [
    {"color": [0, 127, 255], "id": 1, "name": "airplane"},
    {"color": [0, 63, 0], "id": 2, "name": "airport"},
    {"color": [0, 127, 63], "id": 3, "name": "baseball field"},
    {"color": [0, 63, 255], "id": 4, "name": "basketball court"},
    {"color": [0, 0, 127], "id": 5, "name": "bridge"},
    {"color": [0, 127, 127], "id": 6, "name": "chimney"},
    {"color": [0, 0, 63], "id": 7, "name": "expressway service area"},
    {"color": [0, 63, 127], "id": 8, "name": "expressway toll station"},
    {"color": [0, 63, 191], "id": 9, "name": "dam"},
    {"color": [0, 191, 127], "id": 10, "name": "golf field"},
    {"color": [0, 127, 191], "id": 11, "name": "ground track field"},
    {"color": [0, 63, 63], "id": 12, "name": "harbor"},
    {"color": [0, 100, 155], "id": 13, "name": "overpass"},
    {"color": [0, 0, 255], "id": 14, "name": "ship"},
    {"color": [0, 0, 191], "id": 15, "name": "stadium"},
    {"color": [64, 191, 127], "id": 16, "name": "storage tank"},
    {"color": [64, 0, 191], "id": 17, "name": "tennis court"},
    {"color": [128, 63, 63], "id": 18, "name": "train station"},
    {"color": [128, 0, 63], "id": 19, "name": "vehicle"},
    {"color": [191, 63, 0], "id": 20, "name": "windmill"},
    {"color": [127, 63, 0], "id": 21, "name": "soccer ball field"},
    {"color": [63, 255, 0], "id": 22, "name": "roundabout"},
    {"color": [0, 127, 0], "id": 23, "name": "container crane"},
    {"color": [127, 127, 0], "id": 24, "name": "helipad"},
    {"color": [63, 127, 0], "id": 25, "name": "building"},
    {"color": [101, 127, 0], "id": 26, "name": "road"},
    {"color": [127, 191, 0], "id": 27, "name": "water"},
    {"color": [63, 63, 0], "id": 28, "name": "tree"},
    {"color": [100, 155, 0], "id": 29, "name": "grass"},
    {"color": [0, 255, 0], "id": 30, "name": "bareland"},
    {"color": [0, 191, 0], "id": 31, "name": "rangeland"},
    {"color": [191, 127, 64], "id": 32, "name": "developed space"},
    {"color": [0, 191, 64], "id": 33, "name": "agriculture land"},
    {"color": [0, 64, 191], "id": 34, "name": "intersection"},
    {"color": [32, 32, 255], "id": 35, "name": "unlabeled areas, background, clutter"}
]

def _get_OVRSIS95K_meta():
    stuff_ids = [k["id"] for k in OVRSIS95K_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in OVRSIS95K_CATEGORIES]
    stuff_colors = [k["color"] for k in OVRSIS95K_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_OVRSIS95K(root):
    meta = _get_OVRSIS95K_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "OVRSIS95K/train/img", "OVRSIS95K/train/mask"),
        ("val", "OVRSIS95K/val/img", "OVRSIS95K/val/mask"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"OVRSIS95K_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

root = "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets"  # 替换为OVRSIS10K数据集的根目录
register_OVRSIS95K(root)
