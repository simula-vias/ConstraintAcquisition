import os.path as osp

# IOT testing environment
CAServerIPAddress = 'localhost'
CAServerPort = 7044
CAServerInterval = 0.05
LOG_BASE_DIRECTORY = "../benchmarks/queries/"
RUN_NAME = "minigrid"
LOGS_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "logs.txt")
EXAMPLE_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "queries.txt")
GYM_LOGGER_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "logger.txt")
GYM_MONITOR_PATH = osp.join(LOG_BASE_DIRECTORY, RUN_NAME, "monitor.csv")

# Production
# CAServerIPAddress = '192.168.1.107'
# CAServerPort = 7044
# CAServerInterval = 0.05
# LOGS_PATH= "/mnt/d/BigData/MyWork/GitHub/ConstraintAcquisition/benchmarks/queries/minigrid.logs"
# EXAMPLE_PATH = "/mnt/d/BigData/MyWork/GitHub/ConstraintAcquisition/benchmarks/queries/minigrid/minigrid.queries"
# iGYM_LOGGER_PATH = "/mnt/d/BigData/MyWork/GitHub/ConstraintAcquisition/benchmarks/queries/minigrid/minigrid.logger"
# GYM_MONITOR_PATH= "/mnt/d/BigData/MyWork/GitHub/ConstraintAcquisition/benchmarks/queries/minigrid/minigrid.monitor.csv"

#GYM ENVIRONMENT SETUP
GYM_ENVIRONMENT = "MiniGrid-LavaCrossingS9N1-v0"
# GYM_ENVIRONMENT = "MiniGrid-LavaGapS5-v0"
STEPS = 30000
SEED = 20
CARL = "none"  # none/mask/replace