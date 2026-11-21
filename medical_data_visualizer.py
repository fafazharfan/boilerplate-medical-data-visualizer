import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars = ['cardio'], value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = pd.DataFrame(
        df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
        ).rename(columns={'size': 'total'})

    # 7
    cat_plot = sns.catplot(
        x    = 'variable',
        y    = 'total',
        hue  = 'value',
        col  = 'cardio',
        kind = 'bar',
        data = df_cat
        )

    # 8
    fig = cat_plot.figure

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()
    corr = corr.round(1)

    # 13
    mask = corr.mask(np.triu(np.ones(corr.shape)).astype(np.bool))

    # 14
    fig, ax = plt.subplots(figsize = (13,10))

    # 15
    sns.heatmap(mask, annot = True, fmt = '.1f', square = True)

    # 16
    fig.savefig('heatmap.png')
    return fig