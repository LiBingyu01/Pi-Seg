import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

FLAIR_CATEGORIES = [
    {"color": [166, 202, 240], "id": 1, "name": "building"},
    {"color": [128, 128, 0], "id": 2, "name": "pervious surface"},
    {"color": [0, 0, 128], "id": 3, "name": "impervious surface"},
    {"color": [255, 0, 0], "id": 4, "name": "bare soil"},
    {"color": [0, 128, 0], "id": 5, "name": "water"},
    {"color": [128, 0, 0], "id": 6, "name": "coniferous"},
    {"color": [255, 233, 233], "id": 7, "name": "deciduous"},
    {"color": [160, 160, 164], "id": 8, "name": "brushwood"},
    {"color": [0, 128, 128], "id": 9, "name": "vineyard"},
    {"color": [90, 87, 255], "id": 10, "name": "herbaceous vegetation"},
    {"color": [255, 255, 0], "id": 11, "name": "agricultural land"},
    {"color": [255, 192, 0], "id": 12, "name": "plowed land"},
]

def _get_FLAIR_meta():
    stuff_ids = [k["id"] for k in FLAIR_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in FLAIR_CATEGORIES]
    stuff_colors = [k["color"] for k in FLAIR_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_FLAIR(root):
    meta = _get_FLAIR_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("test", "FLAIR_test/image", "FLAIR_test/mask"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"FLAIR_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

root = "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/OVRSIS"  # 替换为FLAIR数据集的根目录
register_FLAIR(root)
