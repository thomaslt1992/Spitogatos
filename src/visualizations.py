import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_median_target_by_group(df, big_group='geography_name', small_group='subtype', target='price_per_sqrm',**kwargs):

    n_groups = len(df[big_group].unique())

    n_rows = int(np.ceil(np.sqrt(n_groups)))
    n_cols = int(np.ceil(n_groups / n_rows))

    if 'fig_size' in kwargs:
        fig_size = [kwargs['fig_size'][0],kwargs['fig_size'][1]]
    else:
        fig_size = [12,7]
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_size[0],fig_size[1]))
    axs = axs.flatten()

    for i, area in enumerate(df[big_group].unique()):
        ppsm_df = df[df[big_group]==area].groupby(small_group)[target].agg(['median', 'std']).reset_index()

        ppsm_df = ppsm_df.sort_values('median', ascending=False)

        sns.set(style="whitegrid")
        sns.barplot(x=small_group, y='median', data=ppsm_df, palette='Blues_r', ax=axs[i])

        axs[i].errorbar(x=ppsm_df[small_group], y=ppsm_df['median'], yerr=ppsm_df['std'], fmt='none', ecolor='black', capsize=3)

        for j, row in ppsm_df.iterrows():
            if np.isfinite(row['median']) and np.isfinite(row['std']):
                axs[i].text(j, row['median'] + row['std'] + 5, f"{row['median']:.0f} +/- {row['std']:.0f}", ha='center', fontsize=8)

        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha='right', fontsize=7)
        axs[i].set_ylabel(target.capitalize(), fontsize=7)
        axs[i].set_title(f'Median {target.capitalize()} by {small_group.capitalize()} in {area}', fontsize=9)

    for i in range(n_groups, n_rows*n_cols):
        fig.delaxes(axs[i])

    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.show()
