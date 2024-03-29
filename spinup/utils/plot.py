import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, no_legend=False, legend_loc='best',
              entropy=False, sigma=False, qbias=False, pretanh=False, kl=False, std=False, color=None, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    # sns.set_palette('bright')

    print("##############")
    # print("##############")
    # xaxis_column_index = data.columns.get_loc(xaxis)
    # value_column_index = data.columns.get_loc(value)

    # data.index += 1
    # data['Epoch'] += 1
    # data = pd.concat([first_row, data])

    # data.iloc[-1] = data.iloc[0]
    # data[xaxis] += 5000
    # data.iloc[-1, column_index] = 0
    #
    # data = data.sort_index()

    # print(data.loc[0])

    sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, legend=(not no_legend), ci='sd', color=color, **kwargs)
    plt.xlabel('environment interactions')
    if entropy:
        plt.ylabel('average entropy values')
    elif sigma:
        plt.ylabel('average sigma values')
    elif qbias:
        if std:
            plt.ylabel('Std of normalized Q bias')
        else:
            plt.ylabel('average normalized Q bias')
       # plt.ylabel('average test return')
    elif pretanh:
        plt.ylabel('max pretanh values')
    elif kl:
        plt.ylabel('scaled average KL Divergence')
        #plt.yscale('symlog')
    else:
        plt.ylabel('average test return')
        #plt.ylabel('average entropy values')
    
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    if not no_legend:
        plt.legend(loc=legend_loc).draggable()

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

def get_datasets(logdir, entropy=False, sigma=False, action_dim=None, qbias=False, pretanh=False, kl=False, std=False, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            if entropy:
                performance = 'AverageLogPi'
            elif sigma:
                performance = 'Std' if 'Std' in exp_data else 'AverageStd'
            elif qbias:
                if std:
                    performance = 'StdNormQBiasAvg' if 'StdNormQBiasAvg' in exp_data else 'StdNormQBias'
                else:
                    performance = 'AverageNormQBiasAvg' #if 'AverageNormQBiasAvg' in exp_data else 'AverageNormQBias'
            elif pretanh:
                performance = 'MaxPreTanh'
            elif kl:
                performance = 'AverageKL'
            else:
                performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns),'Condition1',condition1)
            exp_data.insert(len(exp_data.columns),'Condition2',condition2)
            if entropy:
                exp_data.insert(len(exp_data.columns),'Performance', - exp_data[performance])
                #print (-exp_data[performance])
            elif sigma:
                exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance]/action_dim)
            else:
                exp_data.insert(len(exp_data.columns),'Performance', exp_data[performance])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None, entropy=False,
                     sigma=False, action_dim=None, qbias=False, pretanh=False, kl=False, std=False):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]=='/':
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split('/')[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, entropy, sigma, action_dim, qbias, pretanh, kl, std, leg)
    else:
        for log in logdirs:
            data += get_datasets(log, entropy, sigma, action_dim, qbias, pretanh, kl, std)
    return data


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,  
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', no_legend=False, legend_loc='best',
               save_name=None, xlimit=-1, entropy=None, sigma=None, action_dim=None, qbias=False, pretanh=False, kl=False, std=False, color=None):
    data = get_all_datasets(all_logdirs, legend, select, exclude, entropy, sigma, action_dim, qbias, pretanh, kl, std)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        plt.figure()
        # plt.figure(figsize=(10, 7))
        plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, no_legend=no_legend, legend_loc=legend_loc,
                  estimator=estimator, entropy=entropy, sigma=sigma, qbias=qbias, pretanh=pretanh, kl=kl, std=std, color=color)
        if xlimit > 0:
            plt.xlim(0, xlimit)

    if save_name is not None:
        fig = plt.gcf()
        fig.savefig(save_name)
    else:
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    #parser.add_argument('--xaxis', '-x', default='TotalSteps')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument('--no-legend', action='store_true')
    parser.add_argument('--legend-loc', type=str, default='best')
    parser.add_argument('--save-name', type=str, default=None)
    parser.add_argument('--xlimit', type=int, default=-1)
    parser.add_argument('--entropy', action='store_true')
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--qbias', action='store_true')
    parser.add_argument('--sigma', action='store_true')
    parser.add_argument('--pretanh', action='store_true')
    parser.add_argument('--actdim', type=int, default=1)
    parser.add_argument('--std', action='store_true')
    parser.add_argument('--color', '-color', nargs='*')

    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.
            
        no-legend: if specified then no legend will be shown
        

    """

    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count, 
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est, no_legend=args.no_legend, legend_loc=args.legend_loc, save_name=args.save_name,
               xlimit=args.xlimit, entropy=args.entropy, sigma=args.sigma, action_dim=args.actdim,
               qbias=args.qbias, pretanh=args.pretanh, kl=args.kl, std=args.std, color=args.color)

if __name__ == "__main__":
    main()
