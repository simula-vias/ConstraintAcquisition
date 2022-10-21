import os.path as osp

# IOT testing environment
CAServerIPAddress = 'localhost'
CAServerPort = 7044
CAServerInterval = 0.001
LOG_BASE_DIRECTORY = "../benchmarks/queries/"
# RUN_NAME = "minigrid"
RUN_NAME = "snake"
LOGS_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "logs.csv")
EXAMPLE_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "queries.txt")
GYM_LOGGER_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "sb_logger")
GYM_MONITOR_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "monitor.csv")
SKIP_LOGS_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "skips.csv")

# Production
# CAServerIPAddress = '192.168.1.107'
# CAServerPort = 7044
# CAServerInterval = 0.05
# LOG_BASE_DIRECTORY = "/home/morena/benchmarks/queries/"
# LOGS_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "logs.csv")
# EXAMPLE_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "queries.txt")
# GYM_LOGGER_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "sb_logger")
# GYM_MONITOR_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "monitor.csv")

#GYM ENVIRONMENT SETUP
# GYM_ENVIRONMENT = "MiniGrid-LavaCrossingS11N5-v0"
# GYM_ENVIRONMENT = "MiniGrid-LavaGapS5-v0"
GYM_ENVIRONMENT = "Snake-8x8-v0"
STEPS = 100000
SEED = 20
CARL = "mask"  # none/mask/replace
CA_SERVER_CACHE = 0.01