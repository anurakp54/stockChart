import streamlit as st
import altair as alt
import pandas as pd
from datetime import date, timedelta
import yfinance as yf
import numpy as np
import matpotlib.pyplot as plt
plt.style.use('fivethirtyeight')

st.set_page_config(layout='wide')


simple_analysis = st.container()
special_analysis_1 = st.container()


class stock(object):

    def __init__(self, name, period):
        # Define Hyperparameters

        today = str(date.today())
        start_date = str(date.today() - timedelta(days=period))
        # df = web.DataReader(name, data_source='yahoo', start=start_date, end=today)
        df = yf.download(name, start=start_date, end=today)
        df['Avg Price'] = (df['Adj Close'] + df['High']) * 0.5
        df['Avg Price'] = np.ceil(df['Avg Price'] * 4) / 4
        dfg = df.groupby('Avg Price')["Volume"].sum()
        dfg1 = dfg.reset_index(name='Volume')
        dfg1['Cap'] = dfg1['Avg Price'] * dfg1['Volume']
        mu = dfg1['Cap'].sum() / dfg1['Volume'].sum()
        dfg1['bar'] = (dfg1['Avg Price'] - mu) ** 2 * dfg1['Volume']
        std = np.sqrt(dfg1['bar'].sum() / (dfg1['Volume'].sum() - 1))
        buy_price = mu * 0.9  # allow for 6% margin
        sell_price = mu * 1.2

        self.mu = mu
        self.std = std
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.name = name
        self.data = df
        self.shortEMA = self.data.Close.ewm(span=12, adjust=False).mean()
        self.longEMA = self.data.Close.ewm(span=26, adjust=False).mean()
        self.MACD = self.shortEMA - self.longEMA
        self.signal_line = self.MACD.ewm(span=9, adjust=False).mean()
        self.diff_MACD = (self.MACD - self.signal_line).ewm(span=12, adjust=False).mean()
        self.EMA200 = self.data.Close.ewm(span=200, adjust=False).mean()
        self.EMAVol = self.data.Volume.ewm(span=25, adjust=False).mean()


def signal(signal):
    signal.data['MACD'] = signal.MACD
    signal.data['signal_line'] = signal.signal_line
    signal.data['diff_MACD'] = signal.diff_MACD

    Buy = []
    Sell = []
    flag = -1

    for i in range(0, len(signal.data)):

        if signal.data['diff_MACD'][i] > 0.0 and (i > 26) or signal.buy_price > signal.data['Close'][
            i]:  # Long Trade Algorithm
            # 1. MACD above Signal Line.
            # 2. tenkan_sen (black) above kijun_sen (red)
            # 3. Tenkan_sen higher than Kijun_Sen
            # 4. Chikou_span higher than or cross above the cloud

            if signal.data['tenkan_sen'][i] >= signal.data['kijun_sen'][i] or signal.buy_price > signal.data['Close'][
                i]:  # breakup trend
                Sell.append(np.nan)
                if flag != 1:
                    Buy.append(signal.data['Close'][i])
                    flag = 1
                else:
                    Buy.append(np.nan)

            else:
                Buy.append(np.nan)
                Sell.append(np.nan)

        elif signal.data['diff_MACD'][i - 1] > signal.data['diff_MACD'][i] and (i > 26) or signal.sell_price < \
                signal.data['Close'][i]:

            # Short Trade Algorithm
            # 1. Price must close below the cloud.
            # 2. Cloud ahead must be bearish
            # 3. Tenkan_sen lower than Kijun_sen
            # 4. Chikou_span is below or crosses below the cloud

            if signal.data['tenkan_sen'][i] <= signal.data['kijun_sen'][i] or signal.sell_price < signal.data['Close'][
                i]:  # Stop Loss Sell
                Buy.append(np.nan)
                if flag != 0:
                    Sell.append(signal.data['Close'][i])
                    flag = 0
                else:
                    Sell.append(np.nan)

            # elif  signal.data['MACD'][i] > signal.data['signal_line'][i] and signal.data['tenkan_sen'][i] >= signal.data['kijun_sen'][i] : # Cheap Buy
            #    Sell.append(np.nan)
            #    if flag != 1:
            #        Buy.append(signal.data['Close'][i])
            #        flag = 1
            #    else:
            #        Buy.append(np.nan)

            else:
                Buy.append(np.nan)
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)

    return (Buy, Sell)


def ichimoku(signal):
    nine_high = signal.data['High'].rolling(window=9).max()
    nine_low = signal.data['Low'].rolling(window=9).min()
    twentysix_high = signal.data['High'].rolling(window=26).max()
    twentysix_low = signal.data['Low'].rolling(window=26).min()
    fiftytwo_high = signal.data['High'].rolling(window=52).max()
    fiftytwo_low = signal.data['Low'].rolling(window=52).min()

    signal.data['tenkan_sen'] = (nine_low + nine_high) / 2
    signal.data['kijun_sen'] = (twentysix_low + twentysix_high) / 2

    signal.data['senkou_span_a'] = (0.5 * signal.data['tenkan_sen'] + 0.5 * signal.data['kijun_sen']).shift(26)
    signal.data['senkou_span_b'] = (0.5 * fiftytwo_low + 0.5 * fiftytwo_high).shift(26)

    signal.data['chikou_span'] = signal.data['Close'].shift(-26)

    return (signal)


with st.sidebar:
    choose = st.text_input("Enter Stock Name", "^TNX")
    period = st.slider('Period', 0, 720, value=180, step=30)

with simple_analysis:
    st.title("Stock Graph")
    today = str(date.today())
    st.write(f'Data Date: {today}')
    start_date = str(date.today() - timedelta(days=period))
    # xrate_index = web.get_data_fred('DEXTHUS', start =start_date, end=today)
    # xrate = xrate_index.reset_index()

    choseStock = stock(choose+'.BK', period)
    st.write(f' Current Target Price: {choseStock.mu:.2f}')
    choseStock_df = pd.DataFrame(choseStock.data)
    choseStock_df['Avg Price'] = (choseStock_df['Adj Close'] + choseStock_df['High']) * 0.5
    choseStock_df['Avg Price'] = choseStock_df['Avg Price'].round(2)
    choseStock_df = choseStock_df.reset_index()

    mainplot = alt.Chart(choseStock_df).mark_line().encode(
        alt.X('Date:T', scale=alt.Scale(zero=False)),
        alt.Y('Close:Q', scale=alt.Scale(zero=False))
    ).properties(
        width=800
    )
    line = alt.Chart(pd.DataFrame({'y': [choseStock.mu]})).mark_rule().encode(
        y='y',
        color=alt.value('red')
    )

    choseStock_df['Avg Price'] = np.ceil(choseStock_df['Avg Price'] * 8) / 8
    dfg = choseStock_df.groupby(['Avg Price'], as_index=False)["Volume"].sum()
    histrogram_plot = alt.Chart(dfg).mark_bar().encode(
        x='Avg Price:Q',
        y='Volume:Q'
    ).properties(
        width=800
    )
    histrogram_plot

    """Price Curve versus Volumn"""
    brush = alt.selection_interval()
    price_vol_plot = alt.Chart(choseStock_df).mark_point().encode(
        x='Date:T',
        color=alt.condition(brush, alt.value('green'), alt.value('lightgray'))
    ).properties(
        width=750
    ).add_selection(brush)

    price_vol_plot.encode(alt.Y('Close', scale=alt.Scale(zero=False))) | price_vol_plot.encode(
        alt.Y('Volume', scale=alt.Scale(zero=False)))

    price_plot = alt.Chart(choseStock_df).mark_point().encode(
        alt.X('Date:T'),
        alt.Y('Avg Price:Q', scale=alt.Scale(zero=False)),
        size='Volume:Q'
    ).properties(
        width=900
    )
    """Stock Price with Buying Zone (red line)"""
    mainplot + line
    """Stock Trend with Volume"""
    price_plot + price_plot.transform_loess('Date','Avg Price').mark_line(color='red',size=8)

with special_analysis_1:
    ichimoku_data = ichimoku(choseStock)
    buy_sell_signal = signal(ichimoku_data)
    ichimoku_data.data['Buy_Signal'] = buy_sell_signal[0]
    ichimoku_data.data['Sell_Signal'] = buy_sell_signal[1]

    """Technical Chart"""
    fig, (ax) = plt.subplots(figsize=(20, 10))
    ax.plot(choseStock.data.index.values, choseStock.data['senkou_span_a'], color='black', label="tenkan_sen")
    ax.plot(choseStock.data.index.values, choseStock.data['senkou_span_b'], color='blue', label="kijun_sen")
    ax.plot(df.data.index.values, choseStock.data['MACD'], color='cyan',label='MACD')
    ax.plot(choseStock.data.index.values, choseStock.data['Close'], color='green', label='close')
    ax.scatter(choseStock.data.index.values, choseStock.data['Buy_Signal'], s=200, c="green", alpha=1, marker="v", label="Buy")
    ax.scatter(choseStock.data.index.values, choseStock.data['Sell_Signal'], s=200, c="red", alpha=1, marker="^", label="Sell")
    ax.fill_between(choseStock.data.index.values, y1=choseStock.data['senkou_span_a'], y2=choseStock.data['senkou_span_b'],
                    where=choseStock.data['senkou_span_a'] >=choseStock.data['senkou_span_b'], facecolor='green', interpolate=True,
                    alpha=0.2)
    ax.fill_between(choseStock.data.index.values, y1=choseStock.data['senkou_span_a'], y2=choseStock.data['senkou_span_b'],
                    where=choseStock.data['senkou_span_a'] < choseStock.data['senkou_span_b'], facecolor='red', interpolate=True,
                    alpha=0.2)

    ax.set_title('Ichimoku')
    ax.legend()
    st.pyplot(fig)



def app():
    st.markdown("Use this analytical tool at your discretion!")
