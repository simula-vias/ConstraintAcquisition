import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

# pandas lib
df = pd.read_csv("progress_carl.csv")
print(df.shape)

#add row number
# df['no'] = np.arange(df.shape[0])

#normalize column value between 0..1
# for col in df.columns:
#     if (col in ['episode_len','Skip by CA','Skip By ObsA','CMSize']):
#         df[col] = min_max_scaling(df[col])

plt.plot(df['time/iterations'],df['rollout/ep_len_mean'],'r',label='eval/mean_ep_length')
plt.plot(df['time/iterations'],df['rollout/ep_rew_mean'],'b',label='eval/mean_reward')

# plt.plot(df['no'],df['Skip by CA'],'r',label='skips by CM')
# plt.plot(df['no'],df['Skip By ObsA'],'g',label='skips by Obs')
# plt.plot(df['no'],df['CMSize'],'y',label='CM size')
plt.legend()
plt.show()


df2 = pd.read_csv("progress_rl.csv")
print(df2.shape)
plt.plot(df2['time/iterations'],df2['rollout/ep_len_mean'],'r',label='eval/mean_ep_length')
plt.plot(df2['time/iterations'],df2['rollout/ep_rew_mean'],'b',label='eval/mean_reward')

# plt.plot(df['no'],df['Skip by CA'],'r',label='skips by CM')
# plt.plot(df['no'],df['Skip By ObsA'],'g',label='skips by Obs')
# plt.plot(df['no'],df['CMSize'],'y',label='CM size')
plt.legend()
plt.show()


# plt.fill_between(df['no'],df['episode_len'])
# plt.plot(df['no'],df['Skip by CA'],'r',label='skips by CM')
# plt.plot(df['no'],df['Skip By ObsA'],'g',label='skips by Obs')
# # plt.plot(df['no'],df['CMSize'],'c',label='CM size')
# # plt.plot(df['steps'],df['rewards'])
# plt.title('length of episode')
# plt.legend()
# plt.show()
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
