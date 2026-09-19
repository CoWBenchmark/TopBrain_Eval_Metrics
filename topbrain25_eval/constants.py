from enum import Enum

# TopAneu-36-vessel class for Sep 2026 TA36 submissions
MUL_CLASS_LABEL_MAP = {
    "0": "Background",
    "1": "BA",
    "2": "R-P1P2",
    "3": "L-P1P2",
    "4": "R-ICA-C6-C7",  # new in TA36
    "5": "R-M1",
    "6": "L-ICA-C6-C7",  # new in TA36
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
    "35": "R-ICA-C1-C5",  # new in TA36
    "36": "L-ICA-C1-C5",  # new in TA36
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
SIDEROAD_COMPONENT_LABELS = (
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
# SIDEROAD_COMPONENT_LABELS_CT = SIDEROAD_COMPONENT_LABELS_COMMON + (37, 38, 39)
# SIDEROAD_COMPONENT_LABELS_MR = SIDEROAD_COMPONENT_LABELS_COMMON + (41, 42)

# IoU threshold for detection of "side road" components
# a lenient threshold is set to tolerate more detections
IOU_THRESHOLD = 0.25


# detection results
class DETECTION(Enum):
    TP = "TP"
    TN = "TN"
    FP = "FP"
    FN = "FN"


# for contamination metrics
# To account for uncertainty in boundary delineation, the interface thickness was
# configured to 1 voxel per class (extending into both adjacent FG regions),
# and the surface thickness was set to 5 voxels (mono-directionally shrinking the BG region)
# fgc_threshold of 1% can filter out FGC noise
# underseg_threshold can be more lenient than the fgc_threshold, e.g. 0.10 or 10%
# In this way, we do not flood the FG-by-BG (UnderSeg) metric with false alarms of <10% undersegmentation.
# So two magic numbers for thresholds, one for FG-by-FG (1%) and one for FG-by-BG (10%).
CONTAMINATION_FG_FG_INTERFACE_MARGIN = 1
CONTAMINATION_BG_SURFACE_MARGIN = 5
CONTAMINATION_UNDERSEG_RATIO_THRESH = 0.1
CONTAMINATION_FGC_RATIO_THRESH = 0.01
