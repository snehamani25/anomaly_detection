import seaborn as sns; sns.set(style="ticks", rc={"lines.linewidth": 0.5})
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg',warn=False, force=True)
import matplotlib.pyplot as plt
print(sns.__version__)

EPISODES_COUNT = 20000
TRIM_COUNT = 20000


all_losses = np.load("all_t10/model/losses-all-{}.npy".format(EPISODES_COUNT)) *255*100
all_losses = all_losses[101:]

avenue_losses = np.load("avenue/model/losses.npy-{}.npy".format(EPISODES_COUNT)) *255*100
avenue_losses = avenue_losses[100:]

enter_losses = np.load("enter/model/losses.npy-{}.npy".format(EPISODES_COUNT)) *255* 100
enter_losses = enter_losses[100:]
exit_losses = np.load("exit/model/losses.npy-{}.npy".format(EPISODES_COUNT)) *255*100
exit_losses = exit_losses[100:]
ped1_losses = np.load("ped1/model/losses.npy-{}.npy".format(EPISODES_COUNT)) *255*100
ped1_losses = ped1_losses[100:]
ped2_losses = np.load("ped2/model/losses.npy-{}.npy".format(EPISODES_COUNT)) *255*100
ped2_losses = ped2_losses[100:]


num_rows = 20
data = all_losses
episodes = list(range(0, len(data)))
data_preproc = pd.DataFrame({
    'Episodes': episodes,
    'Avenue': data,
    'All': all_losses,
    'Ped1': ped1_losses,
    'Enter': enter_losses,
    'Exit': exit_losses,
    'Ped2': ped2_losses})


sns.lineplot(x='Episodes', y='value', hue='variable', data=pd.melt(data_preproc, ['Episodes']))
plt.show()
print("Done")
