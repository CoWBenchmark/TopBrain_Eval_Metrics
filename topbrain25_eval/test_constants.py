from topbrain25_eval.constants import (
    MUL_CLASS_LABEL_MAP_COMMON,
    MUL_CLASS_LABEL_MAP_CT,
    MUL_CLASS_LABEL_MAP_MR,
)


def test_MUL_CLASS_LABEL_MAP():
    """
    topbrain
        MR has 42 labels + 1 BG
        CT has 40 labels + 1 BG
        interset 34 labels
    """
    # 34 common labels + 1 background
    assert len(MUL_CLASS_LABEL_MAP_COMMON) == 34 + 1
    # 40 CT labels + 1 background
    assert len(MUL_CLASS_LABEL_MAP_CT) == 40 + 1
    # 42 MR labels + 1 background
    assert len(MUL_CLASS_LABEL_MAP_MR) == 42 + 1

    # topbrain CT is consecutive 0 to 40
    assert list(MUL_CLASS_LABEL_MAP_CT.keys()) == [str(x) for x in range(0, 40 + 1)]

    # topbrain MR is consecutive 0 to 42
    assert list(MUL_CLASS_LABEL_MAP_MR.keys()) == [str(x) for x in range(0, 42 + 1)]
