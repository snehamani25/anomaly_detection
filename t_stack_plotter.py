import seaborn as sns; sns.set(style="ticks", rc={"lines.linewidth": 0.3})
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg',warn=False, force=True)
import matplotlib.pyplot as plt
print(sns.__version__)

EPISODES_COUNT = 20000

all_t5_losses = np.load("all_T5/model/losses.npy-{}.npy".format(EPISODES_COUNT)) *255*100
all_t5_losses = all_t5_losses[100:]

all_t10_losses = np.load("all_t10/model/losses-all-{}.npy".format(EPISODES_COUNT))*255*100
all_t10_losses = all_t10_losses[101:]

all_t20_losses = np.load("all_T20/model/losses-all-{}.npy".format(EPISODES_COUNT))*255*100
all_t20_losses = all_t20_losses[100:]


num_rows = 20
episodes = list(range(0, len(all_t20_losses)))
data_preproc = pd.DataFrame({
    'Episodes': episodes,
    't20': all_t20_losses,
    't5': all_t5_losses,
    't10': all_t10_losses
    })


sns.lineplot(x='Episodes', y='value', hue='variable', data=pd.melt(data_preproc, ['Episodes']))
plt.show()
print("Done")
