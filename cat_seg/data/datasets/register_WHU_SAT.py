import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

WHU_SAT_CATEGORIES = [
    {"color": [192, 192, 192], "id": 1, "name": "background"},
    {"color": [0, 255, 0], "id": 2, "name": "building"},
]

def _get_WHU_SAT_meta():
    stuff_ids = [k["id"] for k in WHU_SAT_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in WHU_SAT_CATEGORIES]
    stuff_colors = [k["color"] for k in WHU_SAT_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_WHU_SAT(root):
    meta = _get_WHU_SAT_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("WHU_SAT", "WHU-SAT/Images", "WHU-SAT/Masks_cvt"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"Building_Extraction_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="tif")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

root = "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/Building_Extraction"  # 替换为WHU_SAT数据集的根目录
register_WHU_SAT(root)
