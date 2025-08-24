from enum import Enum


class TRACK(Enum):
    MR = "mr"
    CT = "ct"


MUL_CLASS_LABEL_MAP_COMMON = {
    "0": "Background",
    "1": "BA",
    "2": "R-P1P2",
    "3": "L-P1P2",
    "4": "R-ICA",
    "5": "R-M1",
    "6": "L-ICA",
    "7": "L-M1",
    "8": "R-Pcom",
    "9": "L-Pcom",
    "10": "Acom",
    "11": "R-A1A2",
    "12": "L-A1A2",
    "13": "R-A3",
    "14": "L-A3",
    "15": "3rd-A2",
    "16": "3rd-A3",
    "17": "R-M2",
    "18": "R-M3",
    "19": "L-M2",
    "20": "L-M3",
    "21": "R-P3P4",
    "22": "L-P3P4",
    "23": "R-VA",
    "24": "L-VA",
    "25": "R-SCA",
    "26": "L-SCA",
    "27": "R-AICA",
    "28": "L-AICA",
    "29": "R-PICA",
    "30": "L-PICA",
    "31": "R-AChA",
    "32": "L-AChA",
    "33": "R-OA",
    "34": "L-OA",
}

# CT label map
MUL_CLASS_LABEL_MAP_CT = MUL_CLASS_LABEL_MAP_COMMON | {
    "35": "VoG",
    "36": "StS",
    "37": "ICVs",
    "38": "R-BVR",
    "39": "L-BVR",
    "40": "SSS",
}

# MR label map
MUL_CLASS_LABEL_MAP_MR = MUL_CLASS_LABEL_MAP_COMMON | {
    "35": "R-ECA",
    "36": "L-ECA",
    "37": "R-STA",
    "38": "L-STA",
    "39": "R-MaxA",
    "40": "L-MaxA",
    "41": "R-MMA",
    "42": "L-MMA",
}

BIN_CLASS_LABEL_MAP = {
    "0": "Background",
    "1": "MergedBin",
}

# NOTE: in case of missing values (FP or FN), set the HD95
# to be roughly the maximum distance of human head = 290 mm
HD95_UPPER_BOUND = 290

# "side road" vessels from topbrain components
# Acom, Pcoms, 3rd-A2, 3rd-A3, PICA, AICA, SCA, OA, AChA
# MMA (MR only)
# BVR, ICV (CT only)
SIDEROAD_COMPONENT_LABELS_COMMON = (
    8,  # R-Pcom
    9,  # L-Pcom
    10,  # Acom
    15,  # 3rd-A2
    16,  # 3rd-A3
    25,  # R-SCA
    26,  # L-SCA
    27,  # R-AICA
    28,  # L-AICA
    29,  # R-PICA
    30,  # L-PICA
    31,  # R-AChA
    32,  # L-AChA
    33,  # R-OA
    34,  # L-OA
)
SIDEROAD_COMPONENT_LABELS_CT = SIDEROAD_COMPONENT_LABELS_COMMON + (37, 38, 39)
SIDEROAD_COMPONENT_LABELS_MR = SIDEROAD_COMPONENT_LABELS_COMMON + (41, 42)

# IoU threshold for detection of "side road" components
# a lenient threshold is set to tolerate more detections
IOU_THRESHOLD = 0.25


# detection results
class DETECTION(Enum):
    TP = "TP"
    TN = "TN"
    FP = "FP"
    FN = "FN"
