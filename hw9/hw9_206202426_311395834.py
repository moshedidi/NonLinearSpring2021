"""
Author:  moshed & eladb
Created on 26/05/2021

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_shares():
    prices_data = pd.read_csv('prices.csv')
    securities_data = pd.read_csv('securities.csv')
    mask = prices_data['date'].apply(lambda x: x[:4] == '2016')
    prices_2016 = prices_data[mask]
    prices_2016[prices_2016.groupby('symbol').count('date') == 252]
    symbols, prices, sectors = [], [[]], []
    return symbols, prices, sectors


def q1():
    load_shares()
    # a TODO: explain in word
    df = pd.read_csv('prices.csv')
    df.head(5)
    # b TODO: explain in word
    mask = df['date'].apply(lambda x: x[:4] == '2016')
    df = df[mask]
    df = df[df['symbol'] == 'AAPL'].reset_index()
    apple_close_prices = df.close
    apple_close_prices.plot()
    # plt.show()
    # c TODO: explain in word


def main():
    q1()


if __name__ == '__main__':
    main()
