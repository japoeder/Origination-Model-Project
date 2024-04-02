import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from matplotlib.ticker import StrMethodFormatter
import warnings
warnings.filterwarnings('ignore')

def histogram_boxplot(data, x=None, hue=None, figsize=(15,5), bins=None, xlabel = None, title = None, font_size=12, hist=True, kde=False, boxplot=True, use_pct=False, xlim=None):

    # Create a figure with one or two subplots depending on boxplot
    if boxplot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 3]})
    else:
        fig, ax2 = plt.subplots(figsize=figsize)

    # Draw boxplot 
    if boxplot:
        sns.boxplot(data=data, x=x, color='peachpuff', ax=ax1)
        ax1.set(xlabel=None)  # remove the redundant x-label

    # Calculate weights if use_pct is True
    weights = None
    if use_pct:
        weights = np.ones_like(data[x])/float(len(data[x]))

    # Plot the histogram with optional KDE
    if hist:
        ax_hist = sns.histplot(data=data, x=x, hue=hue, bins=bins, kde=kde, palette='viridis', ax=ax2, weights=weights)
        ax_hist.set_ylabel('Percent' if use_pct else 'Count')

    # Set labels and title
    if xlabel:
        ax2.set_xlabel(xlabel, fontsize=font_size)
    if title:
        ax2.set_title(title, fontsize=font_size)

    # Set x limits
    if xlim:
        ax2.set_xlim(xlim)

def horizontal_bar(series, avg_col=None, xdim=None, figsize=(12, 4), bins=10, sort_by='y'):

    # Abstract input
    in_df = pd.DataFrame(series)
    feature = series.name

    # If avg_col is provided, concatenate it with in_df
    if avg_col is not None:
        in_df = pd.concat([in_df, avg_col], axis=1)
        avg_col = avg_col.name  # Get the name of the avg_col Series

    # If the series is numeric, create bins
    if np.issubdtype(in_df[feature].dtype, np.number):
        # Create bin edges
        bin_edges = np.linspace(start=in_df[feature].min(), stop=in_df[feature].max(), num=bins+1)

        in_df[feature] = pd.cut(in_df[feature], bins=bin_edges, include_lowest=True)

        # Convert intervals to a string representing the range
        in_df[feature] = in_df[feature].apply(lambda x: f"{x.left:.2f}-{x.right:.2f}")

    # Count the number of occurrences of each category
    if avg_col is None:
        x = in_df.groupby(feature)[feature].count()
        if sort_by == 'x':
            x = x.sort_values()
    else:
        # Compute the average of avg_col for each category
        x = in_df.groupby(feature)[avg_col].mean()
        if sort_by == 'x':
            x = x.sort_values()

    if xdim is not None:
        x = x.tail(xdim)

    ax = x.plot(kind='barh', figsize=figsize, color='#86bf91', zorder=2, width=0.85)

    # Rotate y-axis labels
    for label in ax.get_yticklabels():
        label.set_rotation(45)

    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Switch off ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw vertical axis lines
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Set x-axis label
    ax.set_xlabel("Count" if avg_col is None else "Average of " + avg_col, labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel(feature, labelpad=20, weight='bold', size=12)

    # Format y-axis label
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Switch off ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw vertical axis lines
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Set x-axis label
    ax.set_xlabel("Count" if avg_col is None else "Average of " + avg_col, labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel(feature, labelpad=20, weight='bold', size=12)

    # Format y-axis label
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

def heatmap_boxplot(data, x=None, y=None, hue=None, figsize=(12,7), bins=None, xlabel = None, title = None, font_size=12, hist=True, kde=None, boxplot=True, boxplot_axis=None):

    # If y is specified, set kde to False
    if y is not None and kde is None:
        kde = False

    # If boxplot_axis is not specified, default to x
    if boxplot_axis is None:
        boxplot_axis = x

    # Create a figure with two subplots, adjust the height ratios
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 4]})

    # Draw boxplot 
    if boxplot:
        sns.boxplot(data=data, x=boxplot_axis, color='peachpuff', ax=ax1)
        ax1.set(xlabel=None)  # remove the redundant x-label

    # Plot the histogram with optional KDE
    if hist and hue is not None:
        sns.histplot(data=data, x=x, y=y, hue=hue, bins=bins, kde=kde, palette='viridis', ax=ax2)
    elif hist:
        sns.histplot(data=data, x=x, y=y, bins=bins, kde=kde, ax=ax2)

    # Set labels and title
    if xlabel:
        ax2.set_xlabel(xlabel, fontsize=font_size)
    if title:
        ax1.set_title(title, fontsize=font_size)

def simple_bar(data, x, y, figsize=(12,7), xlabel=None, ylabel=None, title=None, sort_by=None, palette='Blues', n=5):
    # Sort data if sort_by is specified
    if sort_by:
        data = data.sort_values(sort_by)

    plt.figure(figsize=figsize)
    sns.barplot(data=data, x=x, y=y, palette=palette)

    # Reduce number of x-axis labels
    plt.xticks(rotation=45)
    xticks = plt.gca().get_xticks()
    plt.gca().set_xticks(xticks[::n])  # include only every nth xtick

    # Set labels and title
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    plt.show()

def corr_heatmap(data="in_df", num_cols="cols"):
    corr = data[num_cols].corr()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(data.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    return plt.show()
