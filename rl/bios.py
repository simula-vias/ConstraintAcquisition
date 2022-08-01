# IOT testing environment
CAServerIPAddress = 'localhost'
CAServerPort = 7044
CAServerInterval = 0.05
LOGS_PATH= "../benchmarks/queries/minigrid.logs"
EXAMPLE_PATH = "../benchmarks/queries/minigrid/minigrid.queries"
GYM_LOGGER_PATH = "../benchmarks/queries/minigrid/minigrid.logger"
GYM_MONITOR_PATH= "../benchmarks/queries/minigrid/minigrid.monitor.csv"
# Production
# CAServerIPAddress = '192.168.1.107'
# CAServerPort = 7044
# CAServerInterval = 0.05
#LOGS_PATH= "/mnt/d/BigData/MyWork/GitHub/ConstraintAcquisition/benchmarks/queries/minigrid.logs"
#EXAMPLE_PATH = "/mnt/d/BigData/MyWork/GitHub/ConstraintAcquisition/benchmarks/queries/minigrid/minigrid.queries"
#iGYM_LOGGER_PATH = "/mnt/d/BigData/MyWork/GitHub/ConstraintAcquisition/benchmarks/queries/minigrid/minigrid.logger"
#GYM_MONITOR_PATH= "/mnt/d/BigData/MyWork/GitHub/ConstraintAcquisition/benchmarks/queries/minigrid/minigrid.monitor.csv"

GYM_ENVIRONMENT = "MiniGrid-LavaCrossingS11N5-v0"
TIMESTAMP= 30000
MAX_EPISODE_STEPS = 100