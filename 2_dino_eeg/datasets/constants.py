# 采样率
FREQUENCY = 200
# 切片长度1min
# SEGMENT_LEN = 204.8
SEGMENT_LEN = 1800
# 包含的通道
INCLUDED_CHANNELS = [
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "A1",
    "A2",
    "FZ",
    "CZ",
    "PZ",
]

# 最大label+1
N_CLASSES_TUEV = 7

# BCKG CFSZ GNSZ CTSZ ABSZ
# BCKG CBSZ CTSZ ABSZ 尽管只有四个类但是最大label为4，还是使用5（0~4）作为N_CLASSES
N_CLASSES_TUSZ = 5
# BCKG SEIZ
# N_CLASSES_TUSZ = 2

# N_CLASSES_TUAR = 2
N_CLASSES_TUAR = 15

# spindle检测，2分类
N_CLASSES_DREAMS = 2

N_CLASSES_NEONATAL = 1