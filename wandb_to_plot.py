import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb
from scipy.signal import savgol_filter

def get_wandb_data(runs_path, episode_len):
    api = wandb.Api(timeout=29)
    runs = api.runs(path=runs_path)

    metric_list = []
    name_list = []
    for run in runs:
        if 'mujoco' in runs_path:
            metric = run.history(samples=5000, keys=['e_reward'], x_axis='i_episode')
        elif 'metaworld' in runs_path or 'Metaworld' in runs_path:
            metric = run.history(samples=5000, keys=['success_rate'], x_axis='i_episode')
        metric = metric.iloc[:,1].to_list()[:episode_len - 1]
        name = run.name

        metric_list.append(metric)
        name_list.append(name)

    metric_list = np.array(metric_list)
    index = [i+1 for i in range(0,episode_len-1)]
    return pd.DataFrame(data=metric_list.T, index=index, columns=name_list)


def search_data_index(reward_data, keyword=''):
    columns = reward_data.columns.to_list()

    index_ls = []
    for i, idx in enumerate(columns):
        if keyword in idx:
            index_ls.append(i)

    return index_ls


def plot_group_learning_curve(data, fig_name, labels, group_index, smoothing=True, legend_loc=(0.12, -0.45),
                              legend_size=14, dpi=120, xlabel=None, ylabel=None, title=None, ylim=None):

    groups = []
    columns = []
    for i in range(len(group_index)):
        groups.append(data[data.columns[group_index[i]]].values)
        columns.append(data.columns[group_index[i]])

    cmap = plt.cm.get_cmap('Set2', 8)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor='white')
    for d in range(len(labels)):
        data = groups[d]
        dmax = np.max(data, axis=1)
        dmin = np.min(data, axis=1)
        dmean = np.mean(data, axis=1)

        if smoothing:
            y_max = savgol_filter(dmax,31, 3, mode='nearest')
            y_min = savgol_filter(dmin, 31, 3, mode='nearest')
            y = savgol_filter(dmean, 31, 3, mode='nearest')
        else:
            y_max = dmax
            y_min = dmin
            y = dmean

        ax.plot(np.arange(len(data)), y, lw=2.5, color=cmap(d), alpha=0.9, label=labels[d])
        ax.fill_between(x=np.arange(len(data)), y1=y_max, y2=y_min, alpha=0.2, color=cmap(d), linewidth=0)

    ax.grid(lw=0.3)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, len(data))
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.legend(loc=legend_loc, fontsize=legend_size, frameon=0.5)

    fig_path = 'saved_figure/'+fig_name
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    return fig


def plot_single_learning_curve(data, fig_name, labels, single_index, smoothing=True, legend_loc=(0.2, -0.3), xlabel=None, ylabel=None, title=None, ylim=None):

    cmap = plt.cm.get_cmap('Set2', 8)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor='white')
    for d in range(len(labels)):
        y = np.array(data[data.columns[single_index[d]]])

        if smoothing:
            y = savgol_filter(y, 31, 3, mode='nearest')

        ax.plot(np.arange(len(data)), y, lw=2.5, color=cmap(d), alpha=0.9, label=labels[d])

    ax.grid(lw=0.3)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_title(title)
    ax.set_xlim(0, len(data))
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.legend(loc=legend_loc, fontsize=12, frameon=0.5) # fontsize=8

    fig_path = 'saved_figure/'+fig_name
    # plt.savefig(fig_path, dpi=600, bbox_inches='tight')

    # fig.set_figheight(6)
    # fig.set_figwidth(4)
    plt.savefig(fig_path, bbox_inches='tight')


def plot_reward_analysis_plot(reward_data, smoothing=True):
    cmap = plt.cm.get_cmap('Set2', 8)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor='white')
    for d, data in enumerate(reward_data):
        y1 = np.array(data[data.columns[0]])
        y2 = np.array(data[data.columns[1]])
        y1 = savgol_filter(y1, 31, 3, mode='nearest')
        ax.plot(np.arange(len(y1)), y1, linestyle='dashed', lw=2.5, color=cmap(d), alpha=0.9)
        ax.plot(np.arange(len(y2)), y2, lw=2.5, color=cmap(d), alpha=0.9)
    fig.show()


def plot_reward_analysis_single(data, fig_name, labels, smoothing=True, norm_factor=1, legend_loc=(0.44, 0), dpi=120, xlabel=None, ylabel=None, title=None):
    cmap = plt.cm.get_cmap('Set2', 8)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor='white')
    y1 = np.array(data[data.columns[0]])
    y2 = np.array(data[data.columns[1]])
    if smoothing:
        y1= savgol_filter(y1, 31, 3, mode='nearest')
        y2 = savgol_filter(y2, 11, 3, mode='nearest')
    ax.plot(np.arange(len(y1)), y1*norm_factor, lw=2.5, color=cmap(0), alpha=0.9, label=labels[0])
    ax.plot(np.arange(len(y2)), y2, lw=2.5, color=cmap(1), alpha=0.9, label=labels[1])

    ax.grid(lw=0.3)
    ax.set_xlabel(xlabel, labelpad=10, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(0, len(data))
    ax.legend(loc=legend_loc, fontsize=14, frameon=0.5)

    fig_path = 'saved_figure/'+fig_name
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    fig.show()

    return fig


def plot_reward_analysis_scatter(np_data, fig_name, dpi=120, xlabel=None, ylabel=None, title=None):
    cmap = plt.cm.get_cmap('Set2', 8)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor='white')
    ax.scatter(x=np_data[0], y=np_data[1], color=cmap(2))
    ax.set_xlabel(xlabel,labelpad=10,fontsize=16)
    ax.set_ylabel(ylabel,fontsize=16)
    ax.set_title(title,fontsize=16)
    ax.grid(lw=0.3)

    fig_path = 'saved_figure/'+fig_name
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    fig.show()

    return fig


if __name__ == '__main__':
    path = 'OPRRL-Metaworld-ButtonPress-Comparison-Experiments'  #'oprrl_metaworld_button-press-v2_experiment' # oprrl_mujoco_HalfCheetah-v2_human
    reward_data = get_wandb_data(path, 2000)

    # group_index = [
    # [1,2,3],
    # [4,5,6],
    # [7,8,9],
    # ]
    #
    # labels = ['aaa','bbb','ccc']
    #
    # plot_group_learning_curve(reward_data, 'test1.png', labels, group_index, xlabel='episode', ylabel='episode return', title='XXX')

    # plot_single_learning_curve(ant_data, 'ranknoise_Ant.png', labels, index2, xlabel='episode', ylabel='episode return',
    #                            title='Ant with 0.2 scoring noise', legend_loc=(0.22, -0.4))
    # plot_single_learning_curve(reward_data, 'ranknoise_buttonpress.png', labels, index1, xlabel='episode',
    #                            ylabel='success rate', title='ButtonPress with 0.2 scoring noise',
    #                            legend_loc=(0.22, -0.4))

    # reward_data = pd.read_csv('saved_trajs/bp_ra_1.csv')

# def plot_group_learning_curve(data, fig_name, labels, group_index, smoothing=True, legend_loc=(0.12, -0.45),
#                               legend_size=14, dpi=120, xlabel=None, ylabel=None, title=None, ylim=None):

plot_group_learning_curve(reward_data, 'true_user2.pdf', labels, group_index, legend_loc=(0.3,0.), legend_size=10,xlabel='Episode', ylabel='Success Rate', title='ButtonPress', ylim=[0,1],dpi=200)

