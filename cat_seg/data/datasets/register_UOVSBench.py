import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# ==============================================================================
# 1. AquaOV255 Registration
# ==============================================================================
def _get_aqua_ov255_meta():
    # [cite_start]Source: cls_AquaOV255.txt [cite: 1, 2, 3]
    aqua_ov255_classes = [
        "Diver", "Swimmer", "Geoduck", "Linckialaevigata", "Mantaray", "Electricray", 
        "Sawfish", "Bullheadshark", "Greatwhiteshark", "Whaleshark", "Hammerheadshark", 
        "Threshershark", "Seadragon", "Hippocampus", "Morayeel", "Orbicularbatfish", 
        "Lionfish", "Trumpetfish", "Flounder", "Frogfish", "Sailfish", "Enoplosusarmatus", 
        "Pseudanthiaspleurotaenia", "Mola", "Moorishidol", "Bicolorangelfish", 
        "Atlanticspadefish", "Spotteddrum", "Threespotangelfish", "Chromisdimidiata", 
        "Redseabannerfish", "Heniochusvarius", "Maldivesdamselfish", "Scissortailsergeant", 
        "Firegoby", "Twinspotgoby", "Porcupinefish", "Yellowboxfish", "Blackspottedpuffer", 
        "Blueparrotfish", "Stoplightparrotfish", "Pomacentrussulfureus", "Lunarfusilier", 
        "Ocellarisclownfish", "Cinnamonclownfish", "Redseaclownfish", "Pinkanemonefish", 
        "Orangeskunkclownfish", "Giantwrasse", "Spottedwrasse", "Anampsestwistii", 
        "Bluespottedwrasse", "Slingjawwrasse", "Redbreastedwrasse", "Peacockgrouper", 
        "Potatogrouper", "Graysby", "Redmouthgrouper", "Humpbackgrouper", "Coralhind", 
        "Porkfish", "Anyperodonleucogrammicus", "Whitespottedsurgeonfish", 
        "Orangebandsurgeonfish", "Convictsurgeonfish", "Sohalsurgeonfish", "Regalbluetang", 
        "Linedsurgeonfish", "Achillestang", "Powderbluetang", "Whitecheeksurgeonfish", 
        "Saddlebutterflyfish", "Mirrorbutterflyfish", "Bluecheekbutterflyfish", 
        "Blacktailbutterflyfish", "Raccoonbutterflyfish", "Threadfinbutterflyfish", 
        "Eritreanbutterflyfish", "Pyramidbutterflyfish", "Copperbandbutterflyfish", 
        "Giantclams", "Scallop", "Abalone", "Queenconch", "Nautilus", "Tritonstrumpet", 
        "Seaslug", "Dumbooctopus", "Blueringedoctopus", "Commonoctopus", "Squid", 
        "Cuttlefish", "Seaanemone", "Lionsmanejellyfish", "Moonjellyfish", 
        "Friedeggjellyfish", "Fancoral", "Elkhorncoral", "Braincoral", "Seaurchin", 
        "Seacucumber", "Crinoid", "Oreasterreticulatus", "Protoreasternodosus", 
        "Killerwhale", "Spermwhale", "Humpbackwhale", "Seal", "Manatee", "Sealion", 
        "Dolphin", "Walrus", "Dugong", "Turtle", "Snake", "Homarus", "Spinylobster", 
        "Commonprawn", "Mantisshrimp", "Kingcrab", "Hermitcrab", "Cancerpagurus", 
        "Swimmingcrab", "Spannercrab", "Penguin", "Sponge", "Plasticbag", "Plasticbottle", 
        "Plasticcup", "Plasticbox", "Glassbottle", "Surgicalmask", "Tyre", "Can", 
        "Shipwreck", "Wreckedaircraft", "Wreckedcar", "Wreckedtank", "Gun", "Phone", 
        "Ring", "Boots", "Glasses", "Coin", "Statue", "Amphora", "Anchor", "Shipswheel", 
        "Auv", "Rov", "Militarysubmarines", "Personalsubmarines", "Shipsanode", 
        "Overboardvalve", "Propeller", "Seachestgrating", "Submarinepipeline", 
        "Pipelinesanode", "Alligatorgar", "Archerfish", "Arowana", "Banggaicardinalfish", 
        "Barreleyefish", "Baskingshark", "Bigheadcarp", "Blackcarp", "Blanketoctopus", 
        "Bluegill", "Bubblecoral", "Burbot", "Carpsucker", "Catfish", "Chimaera", 
        "Christmastreeworm", "Cleanershrimp", "Clownloach", "Coconutcrab", "Commoncarp", 
        "Conesnail", "Convictcichlid", "Copepod", "Coralshrimp", "Crappie", 
        "Crocodile&alligator", "Cruciancarp", "Cushionstar", "Deepseahatchetfish", 
        "Discusfish", "Fangblenny", "Fangtooth", "Filefish", "Flamingotonguesnail", 
        "Flashlightfish", "Flatworm", "Frilledshark", "Gardeneel", "Giantgourami", 
        "Goblinshark", "Goldfish", "Grasscarp", "Grayling", "Guppy", "Horseshoecrab", 
        "Killifish", "Koi", "Kuhliloach", "Lanternfish", "Largemouthbass", 
        "Leafscorpionfish", "Leafyseadragon", "Mandarinfish", "Marineiguana", 
        "Mimicoctopus", "Mudskipper", "Neontetra", "Oarfish", "Oscarfish", "Paddlefish", 
        "Pearlgourami", "Perch", "Pike", "Pilotfish", "Pineconefish", "Pompomcrab", 
        "Pomacanthusfish", "Pygmyseahorse", "Remora", "Ribboneel", "Rosybarb", "Salmon", 
        "Sanddollar", "Seaangel", "Seaapple", "Seapig", "Seaspider", "Seasquirt", 
        "Silvercarp", "Smallmouthbass", "Snakeheadfish", "Snowcrab", 
        "Spanishdancernudibranch", "Spidercrab", "Spottedgar", "Sturgeon", "Swordtail", 
        "Tigerbarb", "Tilapia", "Triggerfish", "Tripodspiderfish", "Trout", 
        "Velvetbellylanternshark", "Weatherloach", "Wobbegong", "Zebrafish", "Background"
    ]
    return {"stuff_classes": aqua_ov255_classes}

def register_aqua_ov255(root):
    root = os.path.join(root, "AquaOV255") 
    meta = _get_aqua_ov255_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "images", "masks_wb"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"aqua_ov255_{name}_sem_seg"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir, sem_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta)

# ==============================================================================
# 2. DUT-USEG Registration
# ==============================================================================
def _get_dutuseg_meta():
    # [cite_start]Source: cls_dutuseg.txt [cite: 4]
    dutuseg_classes = ["holothurian", "echinus", "scallop", "starfish"]
    return {"stuff_classes": dutuseg_classes}

def register_dutuseg(root):
    root = os.path.join(root, "DUT-USEG")
    meta = _get_dutuseg_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "img", "mask_wb"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"dutuseg_{name}_sem_seg"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir, sem_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta)

# ==============================================================================
# 3. MAS3K Registration
# ==============================================================================
def _get_mas3k_meta():
    # [cite_start]Source: cls_mas3k.txt [cite: 5]
    mas3k_classes = [
        "AngelFish", "BatFish", "Butterflyfish", "ClownFish", "Conch", "Crab", 
        "CrocodileFish", "CuttleFish", "Dolphin", "ElectricRay", "Fish", "Flounder", 
        "FrogFish", "GhostPipeFish", "Grouper", "JellyFish", "LeafySeaDragon", "LionFish", 
        "MantaRay", "MorayEel", "Octopus", "Pagurian", "PipeFish", "Piranha", "RatFish", 
        "ScorpionFish", "SeaCucumber", "SeaHorse", "Seal", "Shark", "Shrimp", "Slug", 
        "StarFish", "Stingaree", "SurgeonFish", "TriggerFish", "Turtle","Background"
    ]
    return {"stuff_classes": mas3k_classes}

def register_mas3k(root):
    root = os.path.join(root, "MAS3K")
    meta = _get_mas3k_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "img", "mask_wb"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"mas3k_{name}_sem_seg"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir, sem_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta)

# ==============================================================================
# 4. SUIM Registration
# ==============================================================================
def _get_suim_meta():
    # [cite_start]Source: cls_suimtrainval.txt [cite: 6]
    suim_classes = [
        "Robots/instruments", "Plants/sea-grass", "Fish and vertebrates", 
        "Human divers", "Reefs and invertebrates", "Wrecks/ruins", 
        "Sand/sea-floor (& rocks)","Background"
    ]
    return {"stuff_classes": suim_classes}

def register_suim(root):
    root = os.path.join(root, "SUIM_train_val")
    meta = _get_suim_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "img", "mask_wb"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"suim_{name}_sem_seg"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='png'))
        MetadataCatalog.get(name).set(image_root=image_dir, sem_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta)

# ==============================================================================
# 5. USIS10K Registration
# ==============================================================================
def _get_usis10k_meta():
    # [cite_start]Source: cls_usis10kvaltest.txt [cite: 7]
    usis10k_classes = [
        "wrecks/ruins", "fish", "reefs", "aquatic plants", 
        "human divers", "robots", "sea-floor","Background"
    ]
    return {"stuff_classes": usis10k_classes}

def register_usis10k(root):
    root = os.path.join(root, "USIS10K_val_test")
    meta = _get_usis10k_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "img", "mask_wb"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"usis10k_{name}_sem_seg"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir, sem_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta)

# ==============================================================================
# 6. USIS16K Registration
# ==============================================================================
def _get_usis16k_meta():
    # [cite_start]Source: cls_usis16k.txt [cite: 8, 9]
    usis16k_classes = [
        "Diver", "Swimmer", "Geoduck", "Linckia laevigata", "Manta ray", "Electric ray", 
        "Sawfish", "Bullhead shark", "Great white shark", "Whale shark", "Hammerhead shark", 
        "Thresher shark", "Sea dragon", "Hippocampus", "Moray eel", "Orbicular batfish", 
        "Lionfish", "Trumpetfish", "Flounder", "Frogfish", "Sailfish", "Enoplosus armatus", 
        "Pseudanthias pleurotaenia", "Mola", "Moorish idol", "Bicolor angelfish", 
        "Atlantic spadefish", "Spotted drum", "Threespot angelfish", "Chromis dimidiata", 
        "Redsea bannerfish", "Heniochus varius", "Maldives damselfish", 
        "Scissortail sergeant", "Fire goby", "Twin-spot goby", "Porcupinefish", 
        "Yellow boxfish", "Blackspotted puffer", "Blue parrotfish", "Stoplight parrotfish", 
        "Pomacentrus sulfureus", "Lunar fusilier", "Ocellaris clownfish", 
        "Cinnamon clownfish", "Red Sea clownfish", "Pink anemonefish", 
        "Orange skunk clownfish", "Giant wrasse", "Spotted wrasse", "Anampses twistii", 
        "Blue-spotted wrasse", "Slingjaw wrasse", "Red-breasted wrasse", "Peacock grouper", 
        "Potato grouper", "Graysby", "Redmouth grouper", "Humpback grouper", "Coral hind", 
        "Porkfish", "Anyperodon leucogrammicus", "Whitespotted surgeonfish", 
        "Orange-band surgeonfish", "Convict surgeonfish", "Sohal surgeonfish", 
        "Regal blue tang", "Lined surgeonfish", "Achilles tang", "Powder blue tang", 
        "Whitecheek surgeonfish", "Saddle butterflyfish", "Mirror butterflyfish", 
        "Bluecheek butterflyfish", "Blacktail butterflyfish", "Raccoon butterflyfish", 
        "Threadfin butterflyfish", "Eritrean butterflyfish", "Pyramid butterflyfish", 
        "Copperband butterflyfish", "Giant clams", "Scallop", "Abalone", "Queen conch", 
        "Nautilus", "Triton's trumpet", "Sea slug", "Dumbo octopus", "Blue-ringed octopus", 
        "Common octopus", "Squid", "Cuttlefish", "Sea anemone", "Lion's mane jellyfish", 
        "Moon jellyfish", "Fried egg jellyfish", "Fan coral", "Elkhorn coral", "Brain coral", 
        "Sea urchin", "Sea cucumber", "Crinoid", "Oreaster reticulatus", 
        "Protoreaster nodosus", "Killer whale", "Sperm whale", "Humpback whale", "Seal", 
        "Manatee", "Sea lion", "Dolphin", "Walrus", "Dugong", "Turtle", "Snake", "Homarus", 
        "Spiny lobster", "Common prawn", "Mantis shrimp", "King crab", "Hermit crab", 
        "Cancer pagurus", "Swimming crab", "Spanner crab", "Penguin", "Sponge", 
        "Plastic bag", "Plastic bottle", "Plastic cup", "Plastic box", "Glass bottle", 
        "Mask", "Tyre", "Can", "Shipwreck", "Wrecked aircraft", "Wrecked car", 
        "Wrecked tank", "Gun", "Phone", "Ring", "Boots", "Glasses", "Coin", "Statue", 
        "Amphora", "Anchor", "Ship's Wheel", "AUV", "ROV", "Military submarines", 
        "Personal submarines", "Ship's anode", "Over board valve", "Propeller", 
        "Sea chest grating", "Submarine pipeline", "Pipeline's anode","Background"
    ]
    return {"stuff_classes": usis16k_classes}

def register_usis16k(root):
    root = os.path.join(root, "USIS16K")
    meta = _get_usis16k_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("all", "images", "masks_wb"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"usis16k_{name}_sem_seg"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir, sem_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta)

# ==============================================================================
# Execution
# ==============================================================================
_root = "/gemini/space/zhaozy/huotao/000OVS/CAT-Seg-UOVS-Cross/datasets"
register_aqua_ov255(_root)
register_dutuseg(_root)
register_mas3k(_root)
register_suim(_root)
register_usis10k(_root)
register_usis16k(_root)