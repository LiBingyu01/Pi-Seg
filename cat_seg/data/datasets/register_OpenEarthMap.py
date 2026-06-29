import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

OpenEarthMap_CATEGORIES = [
    {"color": [128, 128, 128], "id": 1, "name": "Background"},        # 裸地 (灰色/褐色)
    {"color": [128, 128, 128], "id": 2, "name": "Bareland, Barren, Dirt"},        # 裸地 (灰色/褐色)
    {"color": [144, 238, 144], "id": 3, "name": "Pasture, Grass, Lawn"},          # 牧场 (浅绿色)
    {"color": [192, 192, 192], "id": 4, "name": "Developed Space"},  # 已开发空间 (银灰色)
    {"color": [0, 0, 0],       "id": 5, "name": "Roads"},            # 道路 (黑色)
    {"color": [34, 139, 34],   "id": 6, "name": "Tree, Forest"},            # 树木 (深绿色)
    {"color": [0, 0, 255],     "id": 7, "name": "Water, River"},            # 水体 (蓝色)
    {"color": [255, 255, 0],   "id": 8, "name": "Agricultural Land, Cropland"},# 农业用地 (黄色)
    {"color": [255, 0, 0],     "id": 9, "name": "Buildings, Roof, House"}         # 建筑 (红色)
]

def _get_OpenEarthMap_meta():
    stuff_ids = [k["id"] for k in OpenEarthMap_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in OpenEarthMap_CATEGORIES]
    stuff_colors = [k["color"] for k in OpenEarthMap_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_OpenEarthMap(root):
    meta = _get_OpenEarthMap_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "OpenEarthMap/images", "OpenEarthMap/labels"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"OpenEarthMap_{name}_sem_seg"
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

root = "/gemini/space/zhaozy/libingyu/00_OVRSIS_v2/OVRSISBenchV2/datasets/OVRSIS"  # 替换为OpenEarthMap数据集的根目录
register_OpenEarthMap(root)
