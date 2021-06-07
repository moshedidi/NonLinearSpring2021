"""
Author:  moshed & eladb
Created on 26/05/2021

"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt


def pca_project(X, k):
    #### d ####
    X_bar = np.zeros(X.shape[0])
    proj = []
    for j in range(X.shape[1]):
        X_bar[j] = np.average(X[:, j])
    X_centerd = (X.T - X_bar[:, np.newaxis]).T
    XX_T = X_centerd.dot(X_centerd.T)
    eigen = eigs(XX_T, k)[0].real
    vectors = eigs(XX_T, k)[1].real
    sortedEigenVectors = [x for _, x in sorted(zip(eigen, vectors), reverse=True)]
    for vector in sortedEigenVectors:
        for j in range(X.shape[1]):
            proj.append(X_centerd[:, j].dot(vector))
    np.array(proj).reshape((k,X.shape[1]))

    print("good")


def load_shares():
    #### C ###
    prices_data = pd.read_csv('prices.csv')
    securities_data = pd.read_csv('securities.csv')
    mask = prices_data['date'].apply(lambda x: x[:4] == '2016')
    prices_2016 = prices_data[mask]
    checkFullYear = prices_2016.groupby('symbol').count()
    symbols = checkFullYear[checkFullYear['date'] == 252].index
    symbolsDataFrame = symbols.to_frame(index=False)
    sectors = pd.merge(symbolsDataFrame, securities_data, left_on='symbol', right_on='Ticker symbol', how="left")[
        "GICS Sector"]
    dataOnlyFull2016 = pd.merge(symbolsDataFrame, prices_2016, on='symbol', how="inner")
    prices = [df_symbol['close'] for symbol, df_symbol in dataOnlyFull2016.groupby("symbol")]
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
    # q1()
    X = np.array([[0, 0, 0],
                  [3, 3, 5],
                  [6, 8, 6],
                  [9, 7, 9]])
    pca_project(X, 2)


if __name__ == '__main__':
    main()
