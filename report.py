import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table
import os.path
from datetime import datetime, time
import platform
import pandas as pd
import numpy as np
import scipy.stats
from pandas.tools.plotting import autocorrelation_plot
# TODO: possible circular reference, needed to get current version
import eureka
from eureka.backtest import calc_IR, seasonality_metrics, calc_fx_30min_fix_lead_lag

def figures_to_pdf(figures, filename, pdf_info=None):
    with PdfPages(filename) as pdf:
        for i, fig in enumerate(figures):
            if isinstance(fig, matplotlib.axes.Axes):
                fig = fig.get_figure()
            fig.text(8.3/8.5, 0.3/11.0, str(i+1), ha='center', fontsize=10)
            pdf.savefig(fig, transparent=True)

        if pdf_info:
            infodict = pdf.infodict()
            for key, value in pdf_info.items():
                infodict[key] = value

    return pdf


def get_real_name():
    if platform.system() == 'Windows':
        import win32api
        real_name = win32api.GetUserNameEx(3)
        # Comes back in the form "Last, First"
        real_name_parts = [x.strip() for x in reversed(real_name.split(','))]
        real_name = ' '.join(real_name_parts)
        return real_name
    else:
        return None


def create_title_figure(title, rundate, author=None, figsize=None):
    if figsize:
        if abs(figsize[1] / figsize[0] - 0.75) > 0.02:
            import warnings
            warnings.warn('Title template is in 4:3 aspect ratio, for best results adjust your figsize parameter')
    else:
        figsize = (10.5, 8)
    fig = plt.figure(frameon=False, dpi=300, figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    img = plt.imread(os.path.join(os.path.dirname(__file__), 'external', 'report_templates',
                                  'BRAND TEMPLATE_108_highres.png'))
    plt.imshow(img, aspect='auto')

    # TODO: use wrap=True once we're on matplotlib 1.5
    fig.text(0.05, 0.6, title, size=32, fontname='Georgia', weight='bold')

    if author is None:
        author = get_real_name()
    if author is not None:
        fig.text(0.05, 0.5, author, size=14, fontname='Georgia', weight='bold')

    fig.text(0.05, 0.45, rundate.strftime('%B %d, %Y'), size=14, fontname='Calibri')

    return fig


def create_returns_plots(returns):
    fig = plt.figure()
    ax = plt.subplot()
    autocorrelation_plot(returns, ax)

    return fig


def standard_report_pdf(dataset, filename, title, author=None):
    run_time = datetime.now()

    figures = []

    if author is None:
        author = get_real_name()
    title_slide = create_title_figure(title, run_time)
    figures.append(title_slide)

    if 'returns' in dataset:
        returns_plots = create_returns_plots(dataset['returns'])
        figures.append(returns_plots)

    figures_to_pdf(figures, filename, pdf_info={'Title': title,
                                                'Author': get_real_name(),
                                                'CreationDate': run_time,
                                                'EurekaVersion': eureka.__version__})


def _tilt_timing_plot(tilt_timing_cva_df, performance_metrics, ax=None, **kwargs):
    dt_range = '(%s to %s)' % (tilt_timing_cva_df.index[0].strftime('%b-%Y'),
                               tilt_timing_cva_df.index[-1].strftime('%b-%Y'))
    ir = performance_metrics['IR']
    ir_firsthalf = performance_metrics['IR_firsthalf']
    ir_secondhalf = performance_metrics['IR_secondhalf']
    risk = performance_metrics['Risk'] * 100
    turnover = performance_metrics['Turnover']
    pmetrics_summary = '(IR = %.2f, IR_1/IR_2 = %.2f/%.2f, Risk = %.2f%%, Turnover = %0.2f)' % (ir,
                                                                                              ir_firsthalf,
                                                                                              ir_secondhalf,
                                                                                              risk,
                                                                                              turnover)
    fig_title = 'CVA from Tilt & Timing \n%s\n%s' % (pmetrics_summary, dt_range)
    ax = tilt_timing_cva_df.plot(ax=ax, title=fig_title, **kwargs)

    return ax


def _off_the_top_IR_plot(off_the_top_IR_ds, ax=None, **kwargs):
    bar_colors = pd.Series('b', index=off_the_top_IR_ds.index)
    bar_colors.ix['ALL'] = 'r'
    ax = off_the_top_IR_ds.plot(ax=ax, kind='bar', color=list(bar_colors), legend=None, title='Off-the-top IR',
                                **kwargs)

    return ax


def _lead_lag_IR_plot(lead_lag_IR_ds, ax=None, **kwargs):
    bar_colors = pd.Series('b', index=lead_lag_IR_ds.index)
    bar_colors.ix[0] = 'r'
    ax = lead_lag_IR_ds.plot(ax=ax, kind='bar', color=list(bar_colors), legend=None, title='Lead/Lag IR', **kwargs)

    return ax


def _realized_IC_plot(realized_IC_ds, ax=None, **kwargs):
    ax = realized_IC_ds.plot(ax=ax, kind='bar', legend=None, title='Realized ICs', **kwargs)

    return ax

def _highlight_region(ax, start_dt, end_dt, color='red', alpha=0.2):
    # convert datetime index into x coordinates on the given axiss
    xstart = ax.convert_xunits(start_dt)
    xend = ax.convert_xunits(end_dt)
    xmin, xmax = ax.get_xlim()
    if xend < xmin or xstart > xmax:
        # if region is outside the plot than pass
        pass
    else:
        ax.axvspan(xstart, xend, alpha=alpha, facecolor=color)
    return ax
        
def quadrant_plot(backtest_dict, figsize=None, number_of_assets_max_threshold=25, **kwargs):
    """ Quadrant plot for analyzing backtest characteristics. Consists of: tilt/timing, off-the-top IR, lead-lag IR,
    and realized IC plots.

    Parameters
    ----------
    backtest_dict : dict
        Dictionary returned by backtest_metrics or one equivalent
    figsize : tuple, optional
        2-ple of (x, y) plot dimensions in inches
    number_of_assets_max_threshold : int, optional
        Max threshold number of assets allowed for plotting off-the-top IR and realized IC charts.
        If the number of assets exceed the threshold, then these charts are not generated.

    Returns
    -------
    fig : Figure
        Figure containing the four subplots
    """
    plt_flags = [False, False, False, False]

    if 'performance_metrics' in backtest_dict:
        performance_metrics = backtest_dict['performance_metrics']
        if not isinstance(performance_metrics, pd.Series):
            raise ValueError('perf_metrics should be a Series object')

    if 'tilt_timing_cva' in backtest_dict:
        tilt_timing_cva = backtest_dict['tilt_timing_cva']
        if not isinstance(tilt_timing_cva, pd.DataFrame):
            raise ValueError('tilt_timing_cva should be a DataFrame object')
        if 'performance_metrics' in backtest_dict:
            plt_flags[0] = True

    if 'IR_off_the_top' in backtest_dict:
        IR_off_the_top = backtest_dict['IR_off_the_top']
        if not isinstance(IR_off_the_top, pd.Series):
            raise ValueError('IR_off_the_top should be a Series object')
        if IR_off_the_top.shape[0] < number_of_assets_max_threshold:
            plt_flags[1] = True

    if 'IR_lead_lag' in backtest_dict:
        IR_lead_lag = backtest_dict['IR_lead_lag']
        if not isinstance(IR_lead_lag, pd.Series):
            raise ValueError('IR_lead_lag should be a Series object')
        plt_flags[2] = True

    if 'IC_realized' in backtest_dict:
        IC_realized = backtest_dict['IC_realized']
        if not isinstance(IC_realized, pd.Series):
            raise ValueError('IC_realized should be a Series object')
        if IC_realized.shape[0] < number_of_assets_max_threshold:
            plt_flags[3] = True

    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=figsize)

    if plt_flags[0]:
        _tilt_timing_plot(tilt_timing_cva, performance_metrics, ax=ax11, figsize=figsize, **kwargs)
    if plt_flags[1]:
        _off_the_top_IR_plot(IR_off_the_top, ax=ax12, figsize=figsize, **kwargs)
    if plt_flags[2]:
        _lead_lag_IR_plot(IR_lead_lag, ax=ax21, figsize=figsize, **kwargs)
    if plt_flags[3]:
        _realized_IC_plot(IC_realized, ax=ax22, figsize=figsize, **kwargs)
    
    plt.tight_layout()

    return fig

def regime_analysis_plot(stats_df, annualization_factor = 252, figsize=(12, 9), **kwargs):
    """ 
    Plots the characteristics of returns in various regimes/events    

    Parameters
    ----------
    stats_df : DataFrame
        DataFrame containing the characteristics (mean, std, skew etc.) of returns by regimes
    figsize : tuple, optional
        2-ple of (x, y) plot dimensions in inches

    Returns
    -------
    fig : Figure
        Figure containing the six subplots (mean, std, skew, kurtosis, count and t-stat)
    """
    if not isinstance(stats_df, pd.DataFrame):
        raise ValueError('stats_df should be a DataFrame object')

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=figsize)

    # bar plot of mean of returns
    df = stats_df['mean'] * annualization_factor * 100
    df.plot(kind='bar', ax=ax1, **kwargs)
    ax1.legend().set_visible(False)
    ax1.set_title('mean of returns (in %, annualized)')
    ax1.set_xticklabels(df.index, rotation=0)

    # bar plot of std of returns
    df = stats_df['std'] * np.sqrt(annualization_factor) * 100
    df.plot(kind='bar', ax=ax2, **kwargs)
    ax2.legend().set_visible(False)
    ax2.set_title('stdev of returns (in %, annualized)')
    ax2.set_xticklabels(df.index, rotation=0)

    # bar plot of skew of returns
    df = stats_df['skew']
    df.plot(kind='bar', ax=ax3, **kwargs)
    ax3.legend().set_visible(False)
    ax3.set_title('skew of returns')
    ax3.set_xticklabels(df.index, rotation=0)

    # bar plot of kurtosis of returns
    df = stats_df['kurt']
    df.plot(kind='bar', ax=ax4, **kwargs)
    ax4.legend().set_visible(False)
    ax4.set_title('kurtosis of returns')
    ax4.set_xticklabels(df.index, rotation=0)

    # bar plot of sample size
    df = stats_df['count']
    df.plot(kind='bar', ax=ax5, **kwargs)
    ax5.legend().set_visible(False)
    ax5.set_title('sample size')
    ax5.set_xticklabels(df.index, rotation=0)

    # bar plot of t-stat
    df = stats_df['tstat']
    df.plot(kind='bar', ax=ax6, **kwargs)
    ax6.legend().set_visible(False)
    ax6.set_title('t-stat (mean return divided by standard error)')
    ax6.set_xticklabels(df.index, rotation=0)
    
    plt.tight_layout()

    return fig

def histogram(df, bins=None, overlay_normaldist=True, figsize=(12, 9), 
              fontsize=None, xlabel=None, ylabel=None, title=None, **kwargs):
    """
    Plot histogram of the given timeseries with an overlay of normal distribution
    (same mean and stardard deviation as the input data)

    Parameters
    ----------
    df : DataFrame
        Data to be plotted
    bins : int, optional
        Number of bins in the histogram
    overlay_normaldist : bool, optional
        If True, overlay a normal distribution of the same mean and standard deviation as input data
    figsize : tuple, optional
        2-ple of (x, y) dimensions for figures in inches
    fontsize : int, optional
        Font size
    xlabel : string, optional
        Label for x-axis
    ylabel : string, optional
        Label for y-axis
    title : string, optional
        Chart title

    Returns
    -------
    fig : Figure
        Figure containing the table plot
    """
    df = df.dropna().squeeze()
    fig, ax = plt.subplots(figsize=figsize)

    if bins is None:
        n, bins, patches = ax.hist(df.values, normed=True)
    else:
        n, bins, patches = ax.hist(df.values, normed=True, bins=bins)

    # set title and labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # overlay a normal distribution
    if overlay_normaldist:
        ax.plot(bins, mlab.normpdf(bins, df.mean(), df.std()), 'r--', linewidth=2)
        
    # print stats
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    statsstr = 'mean = $%.4f$\nmedian = $%.4f$\nstdev = $%.4f$\nskew = $%.4f$\nkurtosis = $%.4f$' % (df.mean(), 
                                                                   df.median(), df.std(), df.skew(), df.kurtosis())

    ax.text(0.05, 0.95, statsstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    return fig
        
def table_plot(df, figsize=(12, 9), value_format='text', fontsize=None, 
               positive_color=None, negative_color=None, header_color=None, index_color=None, 
               fill_color=None, positive_fill_color=None, negative_fill_color=None, 
               header_fill_color=None, index_fill_color=None, title=None, decimals=2, **kwargs):
    """
    Plot DataFrame as a table

    Parameters
    ----------
    df : DataFrame
        Data to be plotted as a table
    figsize : tuple, optional
        2-ple of (x, y) dimensions for figures in inches
    value_format : string, optional
        Format of the table values. Options supported are 'text' and 'numeric'.
    fontsize : int, optional
        Font size
    positive_color : string, optional
        Color to print positive values
    negative_color : string, optional
        Color to print negative cells
    header_color : string, optional
        Color to print header values
    index_color : string, optional
        Color to print index values
    fill_color : string, optional
        Color to fill table cells
    positive_fill_color : string, optional
        Color to fill cells with positive values
    negative_fill_color : string, optional
        Color to fill cells with negative values
    header_fill_color : string, optional
        Color to fill header cells
    index_fill_color : string, optional
        Color to fill index cells
    title : string, optional
        Table title
    decimals : int, optional
        If value_format is numeric, the number of decimal places used to round values

    Returns
    -------
    fig : Figure
        Figure containing the table plot
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df should be a DataFrame object')    
    if value_format not in ['text', 'numeric']:
        raise ValueError('%s is not a valid format' % value_format)

    # draw table
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    num_cols, num_rows = len(df.columns), len(df.index)
    width, height = 1.0 / num_cols, 1.0 / num_rows

    # add table cells
    loc='center'
    for (i, j), val in np.ndenumerate(df):
        value = df.iloc[i, j]
        text_color = 'black' 
        cell_color = 'none'
        if fill_color is not None:
            cell_color = fill_color            
        if isinstance(value, int) or isinstance(value, float):
            if np.isnan(value):
                continue
            if positive_color is not None and value >= 0:
                text_color = positive_color
            if positive_fill_color is not None and value >= 0:
                cell_color = positive_fill_color
            if negative_color is not None and value < 0:
                text_color = negative_color
            if negative_fill_color is not None and value < 0:
                cell_color = negative_fill_color
            if value_format == 'text':
                tb.add_cell(i + 1, j + 1, width, height, text=str(val), loc=loc, facecolor=cell_color)
            elif value_format == 'numeric':
                value_str = str(round(val, decimals))
                tb.add_cell(i + 1, j + 1, width, height, text=value_str, loc=loc, facecolor=cell_color)
            tb._cells[(i+1, j+1)]._text.set_color(text_color)
        elif isinstance(value, pd.tslib.Timestamp):
            tb.add_cell(i + 1, j + 1, width, height, text=val.strftime('%d-%b-%Y'), loc=loc, facecolor=cell_color)
            tb._cells[(i+1, j+1)]._text.set_color(text_color)
        else:
            tb.add_cell(i + 1, j + 1, width, height, text=str(val), loc=loc, facecolor=cell_color)
            tb._cells[(i+1, j+1)]._text.set_color(text_color)
            
    if index_color is None:
        index_color='black'
    if index_fill_color is not None:
        cell_color = index_fill_color 
    else:
        cell_color = 'none'
    # row labels
    for i, label in enumerate(df.index):
        tb.add_cell(i + 1, 0, width, height, text=label, loc=loc, edgecolor='none', facecolor=cell_color)
        tb._cells[(i+1, 0)]._text.set_color(index_color)

    if header_color is None:
        header_color='black'
    if header_fill_color is not None:
        cell_color = header_fill_color
    else:
        cell_color = 'none'
    # column labels
    for j, label in enumerate(df.columns):
        tb.add_cell(0, j + 1, width, height, text=label, loc=loc, edgecolor='none', facecolor=cell_color)
        tb._cells[(i+1, 0)]._text.set_color(header_color)

    # set font size
    if fontsize is not None:
        tb_cells = tb.properties()['child_artists']
        for cell in tb_cells:
            cell.set_fontsize(fontsize)
    
    # add table to figure        
    ax.add_table(tb)
    
    # set title
    if title is not None:
        if fontsize is not None:
            ax.set_title(title, fontsize=fontsize)
        else:
            ax.set_title(title)
            
    plt.tight_layout()
    
    return fig

def backtest_report(backtest_dict, to_pdf=False, add_title_page=False, title='Backtest Report', author=None,
                    filename='backtest_report.pdf', figsize=None, fontname='Georgia', **kwargs):
    """
    Generate backtest report containing tilt/timing attribution, off-the-top IR, lead-lag IR and ICs

    Parameters
    ----------
    backtest_dict : dict
        Resulting object from eureka.backtest.backtest_metrics or dict with the components specified in Notes section
    to_pdf : bool, optional
        If True, output report to pdf. If False, output report to terminal
    add_title_page : bool, optional
        If True, use SSgA template for adding a title page to the report. If False, output report without the title page
    title : string, optional
        Report title
    author : string, optional
        Author's filename
    filename : string, optional
        Report filename
    figsize : tuple, optional
        2-ple of (x, y) dimensions for figures in inches
    fontname : string, optional
        Font name

    Returns
    -------
    figs : List of Figures
        Backtest quadrant plot, if to_pdf is True. If to_pdf is False, the figure is printed to pdf

    Notes
    -----
    The elements used in the backtest_dict are:

    performance_metrics : Series
        Series containing basic backtest metrics like IRs, turnover etc.
    tilt_timing_cva : DataFrame
        DataFrame containing tilt/timing cumulative value adds
    IR_off_the_top : Series
        Series containing off-the-top IRs
    IR_lead_lag : Series
        Series containing lead/lag IRs
    IC_realized : Series
        Series containing ICs

    """

    if not filename.endswith('.pdf'):
        filename += '.pdf'
    
    # set matplotlib parameters
    plt.rc('font',family=fontname)
    if figsize is None:
        plt.rcParams['figure.figsize'] = (12, 9)
    else:
        plt.rcParams['figure.figsize'] = figsize
    
    figures = []
    run_time = datetime.now()
    if add_title_page:
        title_slide = create_title_figure(title, run_time, author=author, figsize=figsize)
        figures.append(title_slide)

    fig = quadrant_plot(backtest_dict, figsize, **kwargs)
    figures.append(fig)
    if to_pdf:
        figures_to_pdf(figures, filename, pdf_info={'Title': title,
                                                    'Author': get_real_name(),
                                                    'CreationDate': run_time,
                                                    'EurekaVersion': eureka.__version__})
    return figures

def aggregate_report(backtest_dict, add_title_page=True, title='Aggregate Report', author=None,
                    filename='aggregate_report.pdf', figsize=(12,9), fontname='Georgia', fontsize=10, 
                    title_fontsize=15, title_weight='bold', title_offset=0.88, legend_loc='upper left',
                    number_of_assets_max_threshold=25, **kwargs):
    """
    Generate aggregate report (pdf) containing a signal backtest and attribution plots

    Parameters
    ----------
    backtest_dict : dict
        Resulting object from eureka.backtest.backtest_metrics or dict with the components specified in Notes section
    add_title_page : bool, optional
        If True, use SSgA template for adding a title page to the report. If False, output report without the title page
    title : string, optional
        Report title
    author : string, optional
        Author's filename
    filename : string, optional
        Report filename
    figsize : tuple, optional
        2-ple of (x, y) dimensions for figures in inches
    fontname : string, optional
        Font name
    fontsize : int, optional
        Font size
    title_fontsize : int, optional
        Font size for page titles
    title_weight : string, optional
        Weight for page titles    
    title_offset : float, optional
        Offset for displaying page titles
    legend_loc : string, optional
        Legend location        
    number_of_assets_max_threshold : int, optional
        If the number of assets exceed the threshold, then certain plots indicating asset specific metrics (e.g. ICs,
        off-the-top IR etc) are not generated. Also, legend is not generated in plots if the threshold is breached.

    Returns
    -------
    None

    Notes
    -----
    The elements used in the backtest_dict are:

    performance_metrics : Series
        Series containing basic backtest metrics like IRs, turnover etc.
    tilt_timing_cva : DataFrame
        DataFrame containing tilt/timing cumulative value adds
    IR_off_the_top : Series
        Series containing off-the-top IRs
    IR_lead_lag : Series
        Series containing lead/lag IRs
    IC_realized : Series
        Series containing ICs

    """

    if not filename.endswith('.pdf'):
        filename += '.pdf'
    
    kwargs = {'fontsize':fontsize}
    title_parameters = {'fontsize':title_fontsize,
                        'fontname':fontname,
                        'weight':title_weight}
    title_date_range = '(%s to %s)' % (backtest_dict['holdings'].index[0].strftime('%b-%Y'), backtest_dict['holdings'].index[-1].strftime('%b-%Y'))
    title_perf_summary = '(IR = %.2f, Time-Agg IR = %.2f, Risk = %.2f%%, Annual Turnover = %.2f)' % (backtest_dict['performance_metrics']['IR'],
                                                                                               backtest_dict['performance_metrics']['TAIR'],
                                                                                               backtest_dict['performance_metrics']['Risk'] * 100,
                                                                                               backtest_dict['performance_metrics']['Turnover'])
                                                                                              
    # set matplotlib parameters
    plt.rc('font',family=fontname)
    if figsize is None:
        plt.rcParams['figure.figsize'] = (12, 9)
    else:
        plt.rcParams['figure.figsize'] = figsize
    
    figures = []
    run_time = datetime.now()
    
    # Title page
    if add_title_page:
        title_slide = create_title_figure(title, run_time, author=author, figsize=figsize)
        figures.append(title_slide)
    
    # Backtest summary
    fig = quadrant_plot(backtest_dict, figsize, number_of_assets_max_threshold=number_of_assets_max_threshold, **kwargs)
    title = 'Backtest Summary %s' % title_date_range
    fig.suptitle(title, **title_parameters)
    fig.subplots_adjust(top=title_offset)
    figures.append(fig)

    # Signal characteristics : score
    if 'score' in backtest_dict:
        df = backtest_dict['score'].copy()
        del df.index.name
        title = 'Signal Characteristics : Score %s' % title_date_range
        fig, ax = plt.subplots(1)
        if df.shape[1] < number_of_assets_max_threshold:
            df.plot(ax=ax, **kwargs)
            ax.legend(loc=legend_loc, fontsize=fontsize)
        else:
            df.plot(ax=ax, legend=False, **kwargs)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Signal characteristics : alpha
    if 'alpha' in backtest_dict:
        df = backtest_dict['alpha'].copy()
        del df.index.name
        title = 'Signal Characteristics : Alpha %s' % title_date_range
        fig, ax = plt.subplots(1)
        if df.shape[1] < number_of_assets_max_threshold:
            df.plot(ax=ax, **kwargs)
            ax.legend(loc=legend_loc, fontsize=fontsize)
        else:
            df.plot(ax=ax, legend=False, **kwargs)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
 
    # Signal characteristics : holdings
    if 'holdings' in backtest_dict:
        df = backtest_dict['holdings'].copy()
        del df.index.name
        title = 'Signal Characteristics : Holdings %s' % title_date_range
        fig, ax = plt.subplots(1)
        if df.shape[1] < number_of_assets_max_threshold:
            df.plot(ax=ax, **kwargs)
            ax.legend(loc=legend_loc, fontsize=fontsize)
        else:
            df.plot(ax=ax, legend=False, **kwargs)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Signal characteristics : net holdings
    if 'holdings_net' in backtest_dict:
        df = backtest_dict['holdings_net'].copy()
        del df.index.name
        title = 'Signal Characteristics : Net Holdings %s' % title_date_range
        fig, ax = plt.subplots(1)
        df.plot(ax=ax, **kwargs)
        ax.legend(loc=legend_loc, fontsize=fontsize)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
 
    # Performance characteristics : Portfolio Returns Stats
    if 'portfolio_va' in backtest_dict:
        fig = histogram(backtest_dict['portfolio_va'], bins=100, figsize=figsize, xlabel='Portfolio Returns', **kwargs)
        title = 'Performance Characteristics : Portfolio Returns Statistics %s' % title_date_range
        fig.suptitle(title, **title_parameters)
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Performance characteristics : Portfolio Returns Normal Q-Q Plot
    if 'portfolio_va' in backtest_dict:
        portfolio_va = backtest_dict['portfolio_va'].dropna().values.flatten()
        normalized_portfolio_va = (portfolio_va-np.mean(portfolio_va))/np.std(portfolio_va)
        title = 'Performance Characteristics : Portfolio Returns Normal Q-Q Plot %s' % title_date_range
        fig, ax = plt.subplots(1)
        scipy.stats.probplot(normalized_portfolio_va, dist="norm", plot=ax)
        ax.set_title('Normal Q-Q Plot')
        ax.set_xlabel('Normal Quantiles')
        ax.set_ylabel('Actual Quantiles')
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)

     # Performance characteristics : CVA by Asset
    if 'portfolio_va' in backtest_dict and 'va' in backtest_dict and 'cva' in backtest_dict and 'portfolio_cva' in backtest_dict:
        va_df = backtest_dict['va'].copy()
        va_df['Signal'] = backtest_dict['portfolio_va']
        IR_ds = calc_IR(va_df, annualization_factor=backtest_dict['params']['annualization_factor'])
        cols = {}
        for idx in IR_ds.iteritems():
            cols[idx[0]] = idx[0]+' (IR : '+str(round(idx[1],2))+')'
        df = backtest_dict['cva'].copy()
        df['Signal'] = backtest_dict['portfolio_cva']
        df = df.rename(columns=cols)
        del df.index.name
        title = 'Performance Characteristics : CVA by Asset %s\n%s' % (title_date_range,title_perf_summary)
        fig, ax = plt.subplots(1)
        if df.shape[1] < number_of_assets_max_threshold:
            df.plot(ax=ax, **kwargs)
            ax.legend(loc=legend_loc, fontsize=fontsize)
        else:
            df.plot(ax=ax, legend=False, **kwargs)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)

    # Performance characteristics : CVA by Longs and Shorts
    if 'portfolio_long_short_va' in backtest_dict and 'portfolio_long_short_cva' in backtest_dict:
        IR_ds = calc_IR(backtest_dict['portfolio_long_short_va'], annualization_factor=backtest_dict['params']['annualization_factor'])
        cols = {}
        for idx in IR_ds.iteritems():
            cols[idx[0]] = idx[0]+' (IR : '+str(round(idx[1],2))+')'
        df = backtest_dict['portfolio_long_short_cva'].copy()
        df = df.rename(columns=cols)
        del df.index.name
        title = 'Performance Characteristics : CVA by Longs and Shorts %s\n%s' % (title_date_range,title_perf_summary)
        fig, ax = plt.subplots(1)
        df.plot(ax=ax, **kwargs)
        ax.legend(loc=legend_loc, fontsize=fontsize)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)

    # Performance characteristics : CVA by Cross-sectional and Net
    if 'crosssectional_net_va' in backtest_dict and 'crosssectional_net_cva' in backtest_dict:
        IR_ds = calc_IR(backtest_dict['crosssectional_net_va'], annualization_factor=backtest_dict['params']['annualization_factor'])
        cols = {}
        for idx in IR_ds.iteritems():
            cols[idx[0]] = idx[0]+' (IR : '+str(round(idx[1],2))+')'
        df = backtest_dict['crosssectional_net_cva'].copy()
        df = df.rename(columns=cols)
        del df.index.name
        title = 'Performance Characteristics : CVA by Cross-sectional and Net %s\n%s' % (title_date_range,title_perf_summary)
        fig, ax = plt.subplots(1)
        df.plot(ax=ax, **kwargs)
        ax.legend(loc=legend_loc, fontsize=fontsize)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Performance characteristics : IR by year
    if 'IR_by_year' in backtest_dict:
        fig = table_plot(backtest_dict['IR_by_year'], figsize=figsize, negative_color='red', **kwargs)
        title = 'Performance Characteristics : Performance by Year %s' % title_date_range
        fig.suptitle(title, **title_parameters)
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Performance characteristics : Performance by month
    if 'portfolio_va_by_month' in backtest_dict:
        fig = table_plot(backtest_dict['portfolio_va_by_month'], figsize=figsize, value_format='numeric',
                     positive_fill_color='palegreen', negative_fill_color='lightsalmon', **kwargs)
        title = 'Performance Characteristics : Performance by Month (in %%) %s' % title_date_range
        fig.suptitle(title, **title_parameters)
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Performance characteristics : Rolling IR
    if 'IR_rolling' in backtest_dict:
        df = backtest_dict['IR_rolling'].copy()
        del df.index.name
        title = 'Performance Characteristics : Rolling IR %s' % title_date_range
        fig, ax = plt.subplots(1)
        df.plot(ax=ax, **kwargs)
        ax.legend(loc=legend_loc, fontsize=fontsize)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
  
    # Performance characteristics : Rolling IR by Asset
    if 'IR_rolling_assets_1Y' in backtest_dict:
        df = backtest_dict['IR_rolling_assets_1Y'].copy()
        del df.index.name
        title = 'Performance Characteristics : Rolling IR by Asset %s' % title_date_range
        fig, ax = plt.subplots(1)
        if df.shape[1] < number_of_assets_max_threshold:
            df.plot(ax=ax, **kwargs)
            ax.legend(loc=legend_loc, fontsize=fontsize)
        else:
            df.plot(ax=ax, legend=False, **kwargs)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Risk characteristics : Portfolio Ex-ante Risk
    if 'portfolio_exante_risk' in backtest_dict:
        df = backtest_dict['portfolio_exante_risk'].copy() * 100
        del df.index.name
        title = 'Risk Characteristics : Portfolio Ex-Ante Risk (in %%) %s' % title_date_range
        fig, ax = plt.subplots(1)
        df.plot(ax=ax, **kwargs)
        ax.legend(loc=legend_loc, fontsize=fontsize)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)

    # Risk characteristics : Portfolio Ex-ante Risk vs Trailing Ex-post Risk
    if 'portfolio_exante_risk' in backtest_dict and 'trailing_expost_risk' in backtest_dict:
        df = pd.concat([backtest_dict['portfolio_exante_risk'],backtest_dict['trailing_expost_risk']],axis=1) * 100
        del df.index.name
        title = 'Risk Characteristics : Portfolio Ex-Ante Risk vs Ex-Post Risk (in %%) %s' % title_date_range
        fig, ax = plt.subplots(1)
        df.plot(ax=ax, **kwargs)
        ax.legend(loc=legend_loc, fontsize=fontsize)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)

    # Risk characteristics : Asset Risk Contribution
    if 'risk_contribution' in backtest_dict:
        df = backtest_dict['risk_contribution'].copy() * 100
        del df.index.name
        title = 'Risk Characteristics : Asset Risk Contribution (in %%) %s' % title_date_range
        fig, ax = plt.subplots(1)
        if df.shape[1] < number_of_assets_max_threshold:
            df.plot(ax=ax, **kwargs)
            ax.legend(loc=legend_loc, fontsize=fontsize)
        else:
            df.plot(ax=ax, legend=False, **kwargs)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Risk characteristics : Asset Marginal Risk Contribution
    if 'marginal_risk_contribution' in backtest_dict:
        df = backtest_dict['marginal_risk_contribution'].copy()
        del df.index.name
        title = 'Risk Characteristics : Asset Marginal Risk Contribution %s' % title_date_range
        fig, ax = plt.subplots(1)
        if df.shape[1] < number_of_assets_max_threshold:
            df.plot(ax=ax, **kwargs)
            ax.legend(loc=legend_loc, fontsize=fontsize)
        else:
            df.plot(ax=ax, legend=False, **kwargs)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Risk characteristics : Asset Risk Weight
    if 'risk_weight' in backtest_dict:
        df = backtest_dict['risk_weight'].copy()
        del df.index.name
        title = 'Risk Characteristics : Asset Risk Weight %s' % title_date_range
        fig, ax = plt.subplots(1)
        if df.shape[1] < number_of_assets_max_threshold:
            df.plot(ax=ax, **kwargs)
            ax.legend(loc=legend_loc, fontsize=fontsize)
        else:
            df.plot(ax=ax, legend=False, **kwargs)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Risk characteristics : Value-at-Risk
    if 'VaR' in backtest_dict and 'portfolio_va' in backtest_dict:
        df = backtest_dict['VaR'].copy()
        pva_df = backtest_dict['portfolio_va']
        pva_df = pva_df.rename(columns={'Portfolio':'Portfolio VA'})
        df = df.join(pva_df)
        del df.index.name
        title = 'Risk Characteristics : Value-at-Risk (VaR) %s' % title_date_range
        fig, ax = plt.subplots(1)
        df.plot(ax=ax, **kwargs)
        ax.legend(loc=legend_loc, fontsize=fontsize)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Other characteristics : Leverage
    if 'leverage' in backtest_dict:
        df = backtest_dict['leverage'].copy()
        del df.index.name
        title = 'Other Characteristics : Leverage %s' % title_date_range
        fig, ax = plt.subplots(1)
        df.plot(ax=ax, **kwargs)
        ax.legend(loc=legend_loc, fontsize=fontsize)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)

    # Other characteristics : Turnover
    if 'rolling_turnover' in backtest_dict:
        df = backtest_dict['rolling_turnover'].copy()
        del df.index.name
        title = 'Other Characteristics : Turnover %s' % title_date_range
        fig, ax = plt.subplots(1)
        df.plot(ax=ax, **kwargs)
        ax.legend(loc=legend_loc, fontsize=fontsize)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Seasonality characteristics : Seasonality by Month
    if 'portfolio_va' in backtest_dict:
        seasonality_dict = seasonality_metrics(backtest_dict['portfolio_va'])
        title = 'Seasonality Characteristics : Seasonality by Month %s' % title_date_range
        fig = regime_analysis_plot(seasonality_dict['month'], figsize=figsize, **kwargs)
        fig.suptitle(title, **title_parameters)
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)

        # Seasonality characteristics : Seasonality by Quarter
        title = 'Seasonality Characteristics : Seasonality by Quarter %s' % title_date_range
        fig = regime_analysis_plot(seasonality_dict['quarter'], figsize=figsize, **kwargs)
        fig.suptitle(title, **title_parameters)
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
        # Seasonality characteristics : Seasonality by Business Day of Month
        title = 'Seasonality Characteristics : Seasonality by Business Day of Month %s' % title_date_range
        fig = regime_analysis_plot(seasonality_dict['business_day_of_month'], figsize=figsize, **kwargs)
        fig.suptitle(title, **title_parameters)
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
        # Seasonality characteristics : Seasonality by Day of Week
        title = 'Seasonality Characteristics : Seasonality by Day of Week %s' % title_date_range
        fig = regime_analysis_plot(seasonality_dict['day_of_week'], figsize=figsize, **kwargs)
        fig.suptitle(title, **title_parameters)
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Drawdown characteristics : Top 5 Drawdown Periods
    if 'drawdown_periods' in backtest_dict:
        drawdown_table_df = backtest_dict['drawdown_periods'].head(5)
        drawdown_table_df.index = range(1, drawdown_table_df.shape[0] + 1)
        fig = table_plot(drawdown_table_df, figsize=figsize, negative_color='red', **kwargs)
        title = 'Drawdown Characteristics : Top 5 Drawdown Periods %s' % title_date_range
        fig.suptitle(title, **title_parameters)
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    # Drawdown characteristics : Drawdown Profile
    if 'drawdown' in backtest_dict:
        df = backtest_dict['drawdown'].copy() * 100
        del df.index.name
        title = 'Drawdown Characteristics : Drawdown Profile (in %%) (Top 5 highlighted) %s' % title_date_range
        fig, ax = plt.subplots(1)
        df.plot(ax=ax, **kwargs)
        for idx in drawdown_table_df.index:
            ax = _highlight_region(ax,
                               drawdown_table_df.ix[idx, 'From'], 
                               drawdown_table_df.ix[idx, 'To'])
        ax.legend(loc=legend_loc, fontsize=fontsize)
        fig.suptitle(title, **title_parameters)
        plt.tight_layout()
        fig.subplots_adjust(top=title_offset)
        figures.append(fig)
    
    figures_to_pdf(figures, filename, pdf_info={'Title': title,
                                                'Author': get_real_name(),
                                                'CreationDate': run_time,
                                                'EurekaVersion': eureka.__version__})
    return

def fx_30min_fix_lead_lag_report(ret_panel, holdings_df, end_time=time(17,0), timezone='ET', date_freq='B',
                                 add_per_year_charts=True, add_title_page=True, title='FX 30min FIX Lead-Lag Report',
                                 author=None, filename='fx_30min_fix_lead_lag_report.pdf', figsize=(12,9),
                                 fontname='Georgia', fontsize=8, title_fontsize=15, title_weight='bold',
                                 title_offset=0.88, legend_loc='upper left', **kwargs):
    """
    Generate lead-lag IR report (pdf) containing lead-lag IRs for FX 30-min fix returns

    Parameters
    ----------
    ret_panel : Panel
        Panel containing fx 30-min FIX returns (items are FIX times, major_axis is dates and minor_axis is FX assets)
    holdings_df : DataFrame
        DataFrame containing signal holdings for assets
    end_time : datetime.time, optional
        FIX time used to reorder the sequence of lead-lag IRs by FIX timings
    timezone : string, optional
        Timezone of the FIX timings
    date_freq : string, optional
        Frequency of the dates, defaults to business-days
    add_per_year_charts : bool, optional
        If True, add lead-lag charts for each year in the sample.
    add_title_page : bool, optional
        If True, use SSgA template for adding a title page to the report. If False, output report without the title page
    title : string, optional
        Report title
    author : string, optional
        Author's filename
    filename : string, optional
        Report filename
    figsize : tuple, optional
        2-ple of (x, y) dimensions for figures in inches
    fontname : string, optional
        Font name
    fontsize : int, optional
        Font size
    title_fontsize : int, optional
        Font size for page titles
    title_weight : string, optional
        Weight for page titles
    title_offset : float, optional
        Offset for displaying page titles
    legend_loc : string, optional
        Legend location

    Returns
    -------
    None

    """

    if not filename.endswith('.pdf'):
        filename += '.pdf'

    kwargs = {'fontsize':fontsize}
    title_parameters = {'fontsize':title_fontsize,
                        'fontname':fontname,
                        'weight':title_weight}
    title_date_range = '(%s to %s)' % (ret_panel.major_axis[0].strftime('%b-%Y'), ret_panel.major_axis[-1].strftime('%b-%Y'))
    title_time_range = '%s to %s %s' % (ret_panel.items[0].strftime('%H:%M'), ret_panel.items[-1].strftime('%H:%M'), timezone)

    # set matplotlib parameters
    plt.rc('font',family=fontname)
    if figsize is None:
        plt.rcParams['figure.figsize'] = (12, 9)
    else:
        plt.rcParams['figure.figsize'] = figsize

    figures = []
    run_time = datetime.now()

    # Title page
    if add_title_page:
        title_slide = create_title_figure(title, run_time, author=author, figsize=figsize)
        figures.append(title_slide)

    lead_lag_df = calc_fx_30min_fix_lead_lag(ret_panel, holdings_df, end_time)

    # lead/lag
    df = lead_lag_df.copy()
    title = 'Lead/Lag IR : %s %s' % (title_time_range, title_date_range)
    fig, ax = plt.subplots(1)
    df.plot(ax=ax, kind='bar', colormap='jet', legend=False, **kwargs)
    fig.suptitle(title, **title_parameters)
    plt.tight_layout()
    fig.subplots_adjust(top=title_offset)
    figures.append(fig)

    # lag 0 and lag 1
    title = 'Lag 0 and Lag 1 IR %s' % title_date_range
    fig, ax = plt.subplots(1)
    df.ix[0:1].plot(ax=ax, kind='bar', colormap='jet', **kwargs)
    ax.legend(loc=legend_loc, fontsize=fontsize)
    fig.suptitle(title, **title_parameters)
    plt.tight_layout()
    fig.subplots_adjust(top=title_offset)
    figures.append(fig)

    # lead/lag by year
    if add_per_year_charts:
        for year in range(ret_panel.major_axis[0].year,ret_panel.major_axis[-1].year+1):
            if year == ret_panel.major_axis[0].year:
                dt_range = pd.date_range(start=ret_panel.major_axis[0],
                                     end=datetime(year, 12, 31),
                                     freq=date_freq,normalize=True)
            elif year == ret_panel.major_axis[-1].year:
                dt_range = pd.date_range(start=datetime(year, 1, 1),
                                     end=ret_panel.major_axis[-1],
                                     freq=date_freq,normalize=True)
            else:
                dt_range = pd.date_range(start=datetime(year, 1, 1),
                                     end=datetime(year, 12, 31),
                                     freq=date_freq,normalize=True)
            df = calc_fx_30min_fix_lead_lag(ret_panel.reindex(major_axis=dt_range), holdings_df, end_time)

            title = 'Lag 0 and Lag 1 IR (%s)' % str(year)
            fig, ax = plt.subplots(1)
            df.ix[0:1].plot(ax=ax, kind='bar', colormap='jet', **kwargs)
            ax.legend(loc=legend_loc, fontsize=fontsize)
            fig.suptitle(title, **title_parameters)
            plt.tight_layout()
            fig.subplots_adjust(top=title_offset)
            figures.append(fig)

    figures_to_pdf(figures, filename, pdf_info={'Title': title,
                                                'Author': get_real_name(),
                                                'CreationDate': run_time,
                                                'EurekaVersion': eureka.__version__})
    return