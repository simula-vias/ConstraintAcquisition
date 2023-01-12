import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

def average_scaling(series,rows):
    return (np.mean(series[:rows]))
# pandas lib
df = pd.read_csv("logs_avg_carl.csv")
print(df.shape)

#add row number
# df['no'] = np.arange(df.shape[0])


#normalize column value between 0..1
for col in df.columns:
    if (col in ['episode_len','Skip by CA','Skip By ObsA','CMSize','steps','episode_avg_int']):
        df[col] = min_max_scaling(df[col])

plt.bar(df['no'],df['rewards'],label='rewards')
plt.plot(df['no'],df['episode_avg_int'],label='eps_avg')
# plt.plot(df['no'],df['Skip by CA'],'r',label='skips by CM')
# plt.plot(df['no'],df['Skip By ObsA'],'g',label='skips by Obs')
# plt.plot(df['no'],df['steps'])
# plt.plot(df['no'],df['CMSize'],'y',label='CM size')
plt.legend()
plt.show()

plt.fill_between(df['no'],df['episode_len'])
plt.plot(df['no'],df['episode_avg_int'],'r',label='AVG episode length')
# plt.plot(df['no'],df['Skip by CA'],'r',label='skips by CM')
# plt.plot(df['no'],df['Skip By ObsA'],'g',label='skips by Obs')
# plt.plot(df['no'],df['CMSize'],'c',label='CM size')
# plt.plot(df['steps'],df['rewards'])
plt.title('length of episode')
plt.legend()
plt.show()
# pandas lib

#
# #seaborn
# sns.set(style="darkgrid")
#
# sns.kdeplot(df['steps'],df['rewards'])
# plt.show()
# #seaborn


#  __  __                       #
# |  \/  |___ _ _ ___ _ _  __ _ #
# | |\/| / _ \ '_/ -_) ' \/ _` |#
# |_|  |_\___/_| \___|_||_\__,_|#
