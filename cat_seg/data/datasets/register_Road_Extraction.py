import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

Road_Extraction_CATEGORIES = [
    {"color": [192, 192, 192], "id": 1, "name": "background"},
    {"color": [0, 255, 0], "id": 2, "name": "road"},
]

def _get_Road_Extraction_meta():
    stuff_ids = [k["id"] for k in Road_Extraction_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in Road_Extraction_CATEGORIES]
    stuff_colors = [k["color"] for k in Road_Extraction_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_Road_Extraction(root):
    meta = _get_Road_Extraction_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("CHN6_CUG", "CHN6-CUG/image_cvt", "CHN6-CUG/label_cvt"),
        ("DeepGlobe", "DeepGlobe/image_cvt", "DeepGlobe/label_cvt"),
        ("Massachusetts", "Massachusetts/image_cvt", "Massachusetts/label_cvt"),
        ("SpaceNet", "SpaceNet/image_cvt", "SpaceNet/label_cvt"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"Road_Extraction_{name}_sem_seg"
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

root = "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/Road_Extraction"  # 替换为Road_Extraction数据集的根目录
register_Road_Extraction(root)
