import re
import math
import dash
import base64
import numpy as np
import pandas as pd
import matplotlib
import dash_bootstrap_components as dbc
matplotlib.use('Agg')
import seaborn as sns
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.ticker as ticker
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
from matplotlib.ticker import FormatStrFormatter

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.JOURNAL])
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

data = pd.read_csv("vinterdata.csv")
data = data.drop(data.columns[0], axis=1)

exchng = data['exchange'].unique()
symb = data['symbol'].unique()

data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
data['month_year'] = data['datetime'].dt.strftime('%B,%Y')
data['time'] = data['datetime'].dt.strftime('%H')
data['date'] = data['datetime'].dt.date
net_profit = data['close'] - data['open']
return_hour = 100 * net_profit / data['open']
data['hourly_return'] = return_hour


def hourly_return(exc, start_date, end_date, *symbols):
    sym = symbols[0]
    data_n = data.loc[(data['exchange'] == exc) & (data['datetime'] <= end_date) &
                      (data['datetime'] >= start_date) ,['datetime','hourly_return','date','symbol']]
    data_n = data_n[data_n['symbol'].isin(sym)]
    data_n.index = range(len(data_n.index))


    return data_n

def dailyreturn(exc,start_date,end_date, *symbols):
    data_n = data.loc[(data['exchange'] == exc)  & (data['datetime'] <= end_date) & (data['datetime'] >= start_date)
                      & (data['time'] == '23'), ['close', 'date','symbol','datetime']]
    data_n.index = range(len(data_n.index))
    close_val = [None]
    date_val = [None]
    symbol_val = [None]

    for j in symbols[0]:
        data_new = data_n.loc[data_n['symbol'] == j]
        data_new.index = range(len(data_new.index))
        for i in range(0, len(data_new)-1):
            close_val.append((data_new['close'][i + 1] - data_new['close'][i]) * 100 / data_new['close'][i])
            date_val.append(data_new['date'][i+1])
            symbol_val.append(data_new['symbol'][i])

    daily_returns = pd.DataFrame({
        'daily_returns': close_val,
        'datetime': date_val,
        'symbol': symbol_val
    })
    daily_returns['date'] = pd.to_datetime(daily_returns["datetime"]).dt.strftime("%b-%d")
    return daily_returns


def fig_to_uri(in_fig, close_all=True):
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', bbox_inches='tight')
    if close_all and type(in_fig)!=sns.axisgrid.FacetGrid:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)


def get_time(start_date, end_date):
    if start_date is None:
        raise PreventUpdate
    elif start_date is not None:
        start_date = datetime.strptime(re.split('T| ', start_date)[0], '%Y-%m-%d')
        start_date = start_date.strftime('%b %d, %Y')
    if end_date is None:
        raise PreventUpdate
    elif end_date is not None:
        end_date = datetime.strptime(re.split('T| ', end_date)[0], '%Y-%m-%d')
        end_date = end_date.strftime('%b %d, %Y')

    start_date = datetime.strptime(start_date, "%b %d, %Y")
    end_date = datetime.strptime(end_date, "%b %d, %Y")

    return start_date, end_date


def get_corrplot(data_n, colname):
    plt.figure(figsize=(4, 4))
    corr_ = data_n.corr()
    ax = sns.heatmap(
        corr_,
        cmap=sns.light_palette((260, 75, 60), input="husl"),
        annot=True, linewidths=1, cbar_kws={"shrink": .60},
        square=True
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        horizontalalignment='center'
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=360
    )

    ax.set(xlabel=None, ylabel=None)
    ax.set_title(colname)
    plt.tight_layout()
    fig = ax.get_figure()
    out_url = fig_to_uri(fig)
    return out_url


def getdata_corr(start_date, end_date, exc):
    data_n_open = data.loc[(data['exchange'] == exc) & ((data['datetime'] >= start_date) & (
            data['datetime'] <= end_date)), ['open', 'symbol', 'datetime']]
    data_n_open = data_n_open.pivot(index='datetime', columns='symbol', values='open')

    data_n_close = data.loc[(data['exchange'] == exc) & ((data['datetime'] >= start_date) & (
            data['datetime'] <= end_date)), ['close', 'symbol', 'datetime']]
    data_n_close = data_n_close.pivot(index='datetime', columns='symbol', values='close')

    data_n_high = data.loc[(data['exchange'] == exc) & ((data['datetime'] >= start_date) & (
            data['datetime'] <= end_date)), ['high', 'symbol', 'datetime']]
    data_n_high = data_n_high.pivot(index='datetime', columns='symbol', values='high')

    data_n_low = data.loc[(data['exchange'] == exc) & ((data['datetime'] >= start_date) & (
            data['datetime'] <= end_date)), ['low', 'symbol', 'datetime']]
    data_n_low = data_n_low.pivot(index='datetime', columns='symbol', values='low')

    plot_corr_high = get_corrplot(data_n_high, "High")
    plot_corr_low = get_corrplot(data_n_low, "Low")
    plot_corr_open = get_corrplot(data_n_open, "Open")
    plot_corr_close = get_corrplot(data_n_close, "Close")

    return plot_corr_open, plot_corr_close, plot_corr_high, plot_corr_low


def getdata_rollcorr(start_date, end_date, exc, sym, value):
    data_n = data.loc[(data['exchange'] == exc) & ((data['datetime'] >= start_date) & (
            data['datetime'] <= end_date)), [value, 'symbol', 'datetime', 'time']]
    colval = [x.split("|") for x in sym]
    colval = [j for i in colval for j in i]
    data_n = data_n[data_n['symbol'].isin(colval)]
    data_n = data_n.pivot(index='datetime', columns='symbol', values=value)
    data_roll = data_n.rolling(6).corr(pairwise=True)
    data_roll = data_roll.unstack().sort_index(axis=1)
    data_roll = data_roll.dropna()
    data_roll.columns = data_roll.columns.map('|'.join)
    data_roll = data_roll.T.drop_duplicates().T
    for i in data_roll.columns:
        if i in ['btcusd|btcusd', 'ethusd|ethusd', 'btcusd|btcusd', 'ltcusd|ltcusd', 'xrpusd|xrpusd']:
            data_roll = data_roll.drop([i], axis=1)

    data_roll = data_roll.unstack().reset_index()
    data_roll = data_roll.rename(columns={"level_0": "Compare", 0: "values"})
    data_roll['date'] = data_roll['datetime'].dt.date
    data_roll['time'] = data_roll['datetime'].dt.strftime('%H')
    data_roll = data_roll[data_roll['Compare'].isin(sym)]
    return (data_roll)


def daily_volatile(exc, start_date, end_date, *symbols):
    data_n = data.loc[(data['exchange'] == exc) & (data['datetime'] <= end_date) & (data['datetime'] >= start_date) &
                      (data['time'] == '23'), ['close', 'datetime', 'symbol']]
    data_n.index = range(len(data_n.index))
    close_val = [None]
    date_val = [None]
    symbol_val = [None]
    totalvar = [None]

    for j in symbols[0]:
        data_new = data_n.loc[data_n['symbol'] == j]
        data_new.index = range(len(data_new.index))
        total_var = 0
        for i in range(0, len(data_new) - 1):
            sd = math.log(data_new['close'][i + 1]) - math.log(data_new['close'][i])
            var = sd ** 2
            total_var = total_var + var
            totalvar.append(total_var)
            close_val.append(var)
            date_val.append(data_new['datetime'][i + 1])
            symbol_val.append(data_new['symbol'][i + 1])

    daily_volatility = pd.DataFrame({
        'variance': close_val,
        'datetime': date_val,
        'symbol': symbol_val,
        'realized': totalvar
    })

    data_realized = pd.DataFrame(columns=['variance', 'datetime', 'symbol', 'realized'])

    for j in symbols[0]:
        df1 = daily_volatility.loc[daily_volatility['symbol'] == j]
        df1 = df1.iloc[3:]
        data_realized = data_realized.append(df1)

    data_realized['realized'] = [math.sqrt(x) for x in data_realized['realized']]

    return data_realized

tab1 = html.Div([dcc.Dropdown(id='symhist', value='btcusd',
                              className='tab1-content-htmldiv',
                              clearable=False, multi=True, options=[
        {'label': 'btcusd', 'value': 'btcusd'},
        {'label': 'ethusd', 'value': 'ethusd'},
        {'label': 'ltcusd', 'value': 'ltcusd'},
        {'label': 'xrpusd', 'value': 'xrpusd'}]),

                 dcc.Dropdown(id='exchist', value='bf',
                              className='tab2-second-content-htmldiv',
                              clearable=False, options=[
                         {'label': 'bf', 'value': 'bf'},
                         {'label': 'bs', 'value': 'bs'},
                         {'label': 'cb', 'value': 'cb'},
                         {'label': 'ib', 'value': 'ib'},
                         {'label': 'kk', 'value': 'kk'},
                         {'label': 'gi', 'value': 'gi'}
                     ]),
                 html.Div([dcc.Tabs(id='histories', value='Daily', parent_className="secondtab-first-tab",
                                    children=[
                                        dcc.Tab(label='Daily', value='Daily', style={'height': '10px'}),
                                        dcc.Tab(label='Hourly', value='Hourly', style={'height': '10px'})
                                    ])]),
                 html.Div([dcc.Dropdown(id='valuecate', value='open',
                                        className='tab2-openclosehigh-content',
                                        clearable=False, options=[
                         {'label': 'open', 'value': 'open'},
                         {'label': 'close', 'value': 'close'},
                         {'label': 'high', 'value': 'high'},
                         {'label': 'low', 'value': 'low'},
                         {'label': 'Volume Traded', 'value': 'volume'}
                     ])]),

                 html.Div([html.Img(id='tabs-generate-image-histories', src='')], className='figure-display')
                 ], id='tab1')

tab2 = html.Div([dcc.Dropdown(id='sym', value=' ',
                              className='tab-content-htmldiv',
                              clearable=False, multi=True, options=[
        {'label': 'btcusd', 'value': 'btcusd'},
        {'label': 'ethusd', 'value': 'ethusd'},
        {'label': 'ltcusd', 'value': 'ltcusd'},
        {'label': 'xrpusd', 'value': 'xrpusd'}]),

                 dcc.Dropdown(id='exc', value='bf',
                              className='secondslide-content-htmldiv',
                              clearable=False, options=[
                         {'label': 'bf', 'value': 'bf'},
                         {'label': 'bs', 'value': 'bs'},
                         {'label': 'cb', 'value': 'cb'},
                         {'label': 'ib', 'value': 'ib'},
                         {'label': 'kk', 'value': 'kk'},
                         {'label': 'gi', 'value': 'gi'}]),

                 html.Div([dcc.Tabs(id='returns', value='Daily', parent_className="secondtab-first-tab",
                                    children=[
                                        dcc.Tab(label='Daily', value='Daily', style={'height': '10px'}),
                                        dcc.Tab(label='Hourly', value='Hourly', style={'height': '10px'})
                                    ])]),

                html.Div([html.Img(id='tabs-generate-image', src='')], className='figure-display'),
                html.Div([html.Img(id='tabs-generate-image2',src='')],
                         className='figure-display-second')], id='tab2')

tab3 = html.Div([dcc.Dropdown(id='exccorr', value='bf',
                              className='tab3-content-htmldiv',
                              clearable=False, options=[
        {'label': 'bf', 'value': 'bf'},
        {'label': 'bs', 'value': 'bs'},
        {'label': 'cb', 'value': 'cb'},
        {'label': 'ib', 'value': 'ib'},
        {'label': 'kk', 'value': 'kk'},
        {'label': 'gi', 'value': 'gi'}]),

                 html.Div([html.Img(id='tabs-generate-correlation-open', src='')],
                          className='tabs-generate-correlation-open'),
                 html.Div([html.Img(id='tabs-generate-correlation-close', src='')],
                          className='tabs-generate-correlation-close'),
                 html.Div([html.Img(id='tabs-generate-correlation-high', src='')],
                          className='tabs-generate-correlation-high'),
                 html.Div([html.Img(id='tabs-generate-correlation-low', src='')],
                          className='tabs-generate-correlation-low')]
                , id='tab3')

tab4 = html.Div([
    dcc.Dropdown(id='rollcorr', value='btcusd|ethusd',
                 className='tab4-content-rollcorr',
                 clearable=False, multi=True, options=[
            {'label': 'btcusd|ethusd', 'value': 'btcusd|ethusd'},
            {'label': 'btcusd|ltcusd', 'value': 'btcusd|ltcusd'},
            {'label': 'btcusd|xrpusd', 'value': 'btcusd|xrpusd'},
            {'label': 'ethusd|ltcusd', 'value': 'ethusd|ltcusd'},
            {'label': 'ethusd|xrpusd', 'value': 'ethusd|xrpusd'}
        ]), dcc.Dropdown(id='exc_roll', value='bf',
                         className='tab4-second-tab',
                         clearable=False, options=[
            {'label': 'bf', 'value': 'bf'},
            {'label': 'bs', 'value': 'bs'},
            {'label': 'cb', 'value': 'cb'},
            {'label': 'ib', 'value': 'ib'},
            {'label': 'kk', 'value': 'kk'},
            {'label': 'gi', 'value': 'gi'}]),

    html.Div([dcc.Dropdown(id='valuecate_roll', value='open',
                           className='tab4-openclosehigh-content',
                           clearable=False, options=[
            {'label': 'open', 'value': 'open'},
            {'label': 'close', 'value': 'close'},
            {'label': 'high', 'value': 'high'},
            {'label': 'low', 'value': 'low'},
            {'label': 'returns', 'value': 'returns'}
        ])]),
    html.Div([html.Img(id='tabs-rollingcorr', src='')], className='tab4-figure-display')
], id='tab4')

tab5 = html.Div([
    dcc.Dropdown(id='sym_volatile', value='btcusd',
                 className='tab4-content-rollcorr',
                 clearable=False, multi=True, options=[
            {'label': 'btcusd', 'value': 'btcusd'},
            {'label': 'ethusd', 'value': 'ethusd'},
            {'label': 'xrpusd', 'value': 'xrpusd'},
            {'label': 'ltcusd', 'value': 'ltcusd'}
        ]), dcc.Dropdown(id='exc_volatile', value='bf',
                         className='tab4-second-tab',
                         clearable=False, options=[
            {'label': 'bf', 'value': 'bf'},
            {'label': 'bs', 'value': 'bs'},
            {'label': 'cb', 'value': 'cb'},
            {'label': 'ib', 'value': 'ib'},
            {'label': 'kk', 'value': 'kk'},
            {'label': 'gi', 'value': 'gi'}]),
    html.P('Choose date range > 5 day for better visuals',className='para-realizedcorr'),
    html.Div([html.Img(id='generate-volaitlity-image', src='')], className='tab5-figure-display')
], id='tab5')

tabs = dcc.Tabs(id='tabs', value='tab1',
                vertical=True,className='maintab-style',
                children=[
                    dcc.Tab(label='Historical Prices', value='tab1', className='tab_style'),
                    dcc.Tab(label='Returns', value='tab2', className='tab_style'),
                    dcc.Tab(label='Correlation', value='tab3', className='tab_style'),
                    dcc.Tab(label='Rolling Correlation', value='tab4', className='tab_style'),
                    dcc.Tab(label='Realized Volatility', value='tab5', className='tab_style')
                ])
app.layout = html.Div( style={'backgroundColor': colors['background']},
                       children=[html.H1(children='Bitcoin stock analysis', style={'textAlign': 'center',
                                                                                   'color': colors['text']}),
                       tabs,
                       tab1, tab2, tab3, tab4, tab5,
                       html.Div([dcc.DatePickerRange(id='my-date-picker-range',
                                                     display_format='MMM D, YYYY',
                                                     min_date_allowed=datetime(2020, 1, 1),
                                                     max_date_allowed=datetime(2020, 5, 27),
                                                     initial_visible_month=datetime(2020, 1, 12),
                                                     start_date=datetime(2020, 1, 12),
                                                     end_date=datetime(2020, 1, 20)

                                                     )
                                 ], className="secondtab-date-tab"),

                       ])


@app.callback(Output('tab2', 'style'),
              [Input('tabs', 'value')])
def render_content_tab1(tab):
    if tab == 'tab2':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(Output('tab1', 'style'),
              [Input('tabs', 'value')])
def render_content_tab2(tab):
    if tab == 'tab1':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(Output('tab3', 'style'),
              [Input('tabs', 'value')])
def render_content_tab3(tab):
    if tab == 'tab3':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(Output('tab4', 'style'),
              [Input('tabs', 'value')])
def render_content_tab3(tab):
    if tab == 'tab4':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(Output('tab5', 'style'),
              [Input('tabs', 'value')])
def render_content_tab3(tab):
    if tab == 'tab5':
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback([Output('sym', 'multi'), Output('sym', 'value')],
              [Input('returns', 'value')])
def update_dropdown(value):
    if value == 'Daily':
        return [True, 'btcusd']
    elif value == 'Hourly':
        return [False, "ethusd"]


@app.callback([Output('tabs-generate-image', component_property='src'),
              Output('tabs-generate-image2', component_property='src'),
              Output('tabs-generate-image2', 'style')],
              [Input('sym', 'value'),
               Input('sym', 'multi'),
               Input('exc', 'value'),
               Input('my-date-picker-range', 'start_date'),
               Input('my-date-picker-range', 'end_date'),
               Input('tabs', 'value'),
               Input('returns', 'value')
               ])
def update_returns(sym, multi, exc, start_date, end_date, tab,returns):
    if tab == 'tab2':
        gettime = get_time(start_date, end_date)
        sns.set_style("darkgrid")
        plt.figure(figsize=(10, 3))
        start_date = gettime[0]
        end_date = gettime[1]
        if returns =="Daily":
            if not sym:
                raise PreventUpdate
            else:
                if type(sym) == str:
                    sym = [sym]
                daily_returns = dailyreturn(exc, start_date, end_date, sym)
                daily_returns = daily_returns.dropna()
                ax = sns.lineplot(data=daily_returns, x='datetime', y='daily_returns', hue='symbol', sort=False)
                labels = [str(i) + "%" for i in ax.get_yticks()]
                ax.set_yticklabels(labels)
                plt.xticks(rotation=90)
                ax.set(xlabel=None, ylabel='Daily returns')
                diff_dates = abs((end_date - start_date).days)
                if (diff_dates >= 3 and diff_dates <= 10):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.8))
                elif (diff_dates <= 3):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
                elif (diff_dates > 10):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1 * diff_dates))
                daily_hist = daily_returns
                if sym:
                    ghist = sns.FacetGrid(daily_hist, col="symbol", col_wrap=2, hue='symbol',sharex=False,sharey=False)
                    ghist.map(plt.hist, "daily_returns")
                    ghist.fig.suptitle("Histogram of daily returns",x=0.3,y=1)
                    j=0
                    symb = sym
                    if exc=='ib':
                        if 'xrpusd' in symb :
                            symb.remove('xrpusd')
                        if 'ltcusd' in symb:
                            symb.remove('ltcusd')

                    for i in symb:
                        ghist.axes[j].set_xlabel(i)
                        ghist.axes[j].set_title(" ")
                        j=j+1
                    out_url2 = fig_to_uri(ghist)


                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.15, 0.7), loc='right')
                plt.tight_layout()
                fig = ax.get_figure()
                out_url = fig_to_uri(fig)
                return [out_url,out_url2,{'display': 'block'}]
        elif returns=="Hourly":
            if not sym:
                raise PreventUpdate
            else:
                if type(sym) == str:
                    sym = [sym]

                hourly = hourly_return(exc, start_date, end_date, sym)

                ax2 = sns.lineplot(data=hourly, x='datetime', y='hourly_return', hue='symbol', sort=False)
                labels = [str(i) + "%" for i in ax2.get_yticks()]
                ax2.set_yticklabels(labels)
                plt.xticks(rotation=90)
                ax2.set(xlabel=None, ylabel='Hourly returns')
                diff_dates = abs((end_date - start_date).days)
                if (diff_dates >= 3 and diff_dates <= 10):
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.8))
                elif (diff_dates <= 3):
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
                elif (diff_dates > 10):
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.1 * diff_dates))
                daily_hist = hourly
                if sym:
                    ghist2 = sns.FacetGrid(daily_hist, col="symbol", col_wrap=2, hue='symbol', sharex=False, sharey=False)
                    ghist2.map(plt.hist, "hourly_return")
                    ghist2.fig.suptitle("Histogram of hourly returns", x=0.3, y=1)
                    j = 0
                    symb = sym
                    if exc == 'ib':
                        if 'xrpusd' in symb:
                            symb.remove('xrpusd')
                        if 'ltcusd' in symb:
                            symb.remove('ltcusd')

                    for i in symb:
                        ghist2.axes[j].set_xlabel(i)
                        ghist2.axes[j].set_title(" ")
                        j = j + 1
                    out_url2 = fig_to_uri(ghist2)

                handles, labels = ax2.get_legend_handles_labels()
                ax2.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.15, 0.7), loc='right')
                plt.tight_layout()
                fig = ax2.get_figure()
                out_url = fig_to_uri(fig)
                return [out_url,out_url2,{'display': 'block'}]
            #clrs = ['darkorchid' if x > 2 or x < -2 else 'lightcoral' for x in hourly['hourly_return']]

            # ax = sns.barplot(data=hourly, x='datetime', y='hourly_return', palette=clrs)
            # labels = [str(i) + "%" for i in ax.get_yticks()]
            # ax.set_yticklabels(labels)
            # labels = list()
            #
            # for i in ax.get_xticklabels():
            #     labels.append(datetime.strptime(i.get_text()[:10], '%Y-%m-%d'))
            #
            # new_labels = list()
            # y = ax.get_xticks()[::24]
            #
            # for i in y:
            #     new_labels.append(labels[i].strftime('%b-%d'))
            #
            # ax.xaxis.set_ticks(y)
            # ax.set_xticklabels(new_labels, rotation=90)
            # ax.set(xlabel=None, ylabel='Hourly returns')
            #
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.15, 0.7), loc='right')
            plt.tight_layout()
            fig = ax.get_figure()
            out_url = fig_to_uri(fig)
            return out_url
    else:
        raise PreventUpdate



@app.callback(Output('tabs-generate-image-histories', component_property='src'),
              [Input('symhist', 'value'),
               Input('exchist', 'value'),
               Input('my-date-picker-range', 'start_date'),
               Input('my-date-picker-range', 'end_date'),
               Input('tabs', 'value'),
               Input('valuecate', 'value'),
               Input('histories', 'value')
               ])
def update_histories(symhist, exchist, start_date, end_date, tab, col_value, hist):
    if tab == 'tab1' and len(symhist) != 0:
        gettime = get_time(start_date, end_date)

        sns.set_style("darkgrid")
        plt.figure(figsize=(10, 3))
        start_date = gettime[0]
        end_date = gettime[1]
        if type(symhist) == str:
            symhist = [symhist]
        colval = col_value
        col_value = [col_value, 'datetime', 'symbol']
        if hist == "Hourly":
            data_n = data.loc[(data['exchange'] == exchist) & (data['datetime'] >= start_date) & (
                    data['datetime'] <= end_date), col_value]

        if hist == "Daily":
            data_n = data.loc[
                (data['exchange'] == exchist) & (data['time'] == '00') & ((data['datetime'] >= start_date) & (
                        data['datetime'] <= end_date)), col_value]

        data_n = data_n[data_n['symbol'].isin(symhist)]

        ax = sns.lineplot(data=data_n, x='datetime', y=colval, hue='symbol', ci=None)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.15, 0.7), loc='right')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        if (colval) == "volume" and 'xrpusd' in symhist:
            ax.yaxis.offsetText.set_visible(False)
            ax.set(xlabel='', ylabel='Historical prices (1E+6)')
        else:
            ax.set(xlabel='', ylabel='Historical prices')

        plt.tight_layout()
        fig = ax.get_figure()
        out_url = fig_to_uri(fig)
        return out_url


@app.callback([Output('tabs-generate-correlation-open', component_property='src'),
               Output('tabs-generate-correlation-close', component_property='src'),
               Output('tabs-generate-correlation-high', component_property='src'),
               Output('tabs-generate-correlation-low', component_property='src')],
              [Input('tabs', 'value'),
               Input('exccorr', 'value'),
               Input('my-date-picker-range', 'start_date'),
               Input('my-date-picker-range', 'end_date')])
def update_correlation(tab, exc, start_date, end_date):
    if tab == 'tab3':
        gettime = get_time(start_date, end_date)
        start_date = gettime[0]
        end_date = gettime[1]
        corr_plot = getdata_corr(start_date, end_date, exc)
        return corr_plot

    else:
        raise PreventUpdate


@app.callback(Output('tabs-rollingcorr', component_property='src'),
              [Input('tabs', 'value'),
               Input('rollcorr', 'value'),
               Input('exc_roll', 'value'),
               Input('valuecate_roll', 'value'),
               Input('my-date-picker-range', 'start_date'),
               Input('my-date-picker-range', 'end_date')])
def update_rolling(tab, sym, exc_roll, value, start_date, end_date):
    if tab == 'tab4':
        gettime = get_time(start_date, end_date)
        start_date = gettime[0]
        end_date = gettime[1]
        plt.figure(figsize=(10, 4))
        if type(sym) == str:
            sym = [sym]
        data_roll = getdata_rollcorr(start_date, end_date, exc_roll, sym, value)
        ax = sns.lineplot(data=data_roll, x='datetime', y='values', hue='Compare')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.25, 0.7), loc='right')
        diff_dates = abs((end_date - start_date).days)
        if (diff_dates >= 3 and diff_dates <= 10):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        elif (diff_dates <= 3):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H hrs'))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        elif (diff_dates > 10):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1 * diff_dates))
        plt.title('6 hours rolling correlation of bitcoins', y=1, x=0.2, fontsize=10)
        ax.set(xlabel=None, ylabel='Correlation')
        ax.xaxis.set_tick_params(rotation=90)
        fig = ax.get_figure()
        out_url = fig_to_uri(fig)
        return out_url
    else:
        raise PreventUpdate


@app.callback(Output('generate-volaitlity-image', component_property='src'),
              [Input('sym_volatile', 'value'),
               Input('exc_volatile', 'value'),
               Input('my-date-picker-range', 'start_date'),
               Input('my-date-picker-range', 'end_date'),
               Input('tabs', 'value')
               ])
def update_returns(sym, exc, start_date, end_date, tab):
    gettime = get_time(start_date, end_date)
    start_date = gettime[0]
    end_date = gettime[1]
    diff_dates = abs((end_date - start_date).days)
    if tab == 'tab5' and diff_dates>3:
        sns.set_style("darkgrid")
        plt.figure(figsize=(10, 3))
        if type(sym) == str:
            sym = [sym]
        daily_returns = daily_volatile(exc, start_date, end_date, sym)
        daily_returns = daily_returns.dropna()

        ax = sns.lineplot(data=daily_returns, x='datetime', y='realized', hue='symbol', sort=False)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.15, 0.7), loc='right')
        if (diff_dates >= 3 and diff_dates <= 10):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.8))
        elif (diff_dates > 10):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1 * diff_dates))

        ax.set(xlabel=None, ylabel='Realized Correlation')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        plt.title('5 days realized volatility of bitcoins', y=1, x=0.2, fontsize=10)
        plt.tight_layout()
        fig = ax.get_figure()
        out_url = fig_to_uri(fig)
        return out_url
    else:
        raise PreventUpdate



if __name__ == '__main__':
    app.run_server(debug=True)
