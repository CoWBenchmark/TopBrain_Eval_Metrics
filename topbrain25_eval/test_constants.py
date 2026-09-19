from topbrain25_eval.constants import MUL_CLASS_LABEL_MAP, SIDEROAD_COMPONENT_LABELS


def test_MUL_CLASS_LABEL_MAP():
    """
    topbrain
        TA36 =  topbrainv1-common-34 labels + 2 infraICA
    """
    # topbrainv1 34 common labels + 1 background + 2 infraICA
    assert len(MUL_CLASS_LABEL_MAP) == 34 + 1 + 2

    # topbrain TA36 is consecutive 0 to 36
    assert list(MUL_CLASS_LABEL_MAP.keys()) == [str(x) for x in range(0, 36 + 1)]


def test_SIDEROAD_COMPONENT_LABELS():
    # M1 is highway
    label = "L-M1"
    key = int(next(k for k, v in MUL_CLASS_LABEL_MAP.items() if v == label))
    assert key not in SIDEROAD_COMPONENT_LABELS

    # PICA is sideroad
    label = "L-PICA"
    key = int(next(k for k, v in MUL_CLASS_LABEL_MAP.items() if v == label))
    assert key in SIDEROAD_COMPONENT_LABELS

    # A3 is highway
    label = "R-A3"
    key = int(next(k for k, v in MUL_CLASS_LABEL_MAP.items() if v == label))
    assert key not in SIDEROAD_COMPONENT_LABELS

    # OA is sideroad
    label = "R-OA"
    key = int(next(k for k, v in MUL_CLASS_LABEL_MAP.items() if v == label))
    assert key in SIDEROAD_COMPONENT_LABELS
