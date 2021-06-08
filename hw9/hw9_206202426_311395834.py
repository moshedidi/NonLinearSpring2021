"""
Author:  moshed & eladb
Created on 26/05/2021

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigs


# def pca_project(X, k):
#     """ Q1 D """
#     X = np.array(X)
#     X_bar = np.zeros(X.shape[1])
#     proj = []
#     for j in range(X.shape[1]):
#         X_bar[j] = np.average(X[:, j])
#     X_centerd = X - X_bar
#     XX_T = np.dot(X_centerd,X_centerd.T)
#     eigen = eigs(XX_T, k)[0].real
#     vectors = eigs(XX_T, k)[1].real
#     sortedEigenVectors = [x for _, x in sorted(zip(eigen, vectors.T), reverse=True)]
#     for j in range(X.shape[1]):
#         for vector in sortedEigenVectors:
#             proj.append(X_centerd[:, j].dot(vector))
#     proj = np.array(proj).reshape((k, X.shape[1]),order='F')
#     return proj

def pca_project(X, k):
    """ Q1 D """
    X = np.array(X)
    proj = []
    X_centerd = X - np.mean(X, axis=0)
    XTX = np.dot(X_centerd.T,X_centerd)
    eigen = eigs(XTX, k)[0].real
    vectors = eigs(XTX, k)[1].real
    sortedEigenVectors = [x for _, x in sorted(zip(eigen, vectors.T), reverse=True)]
    for i in range(X.shape[0]):
        for vector in sortedEigenVectors:
            proj.append(X_centerd[i, :].dot(vector))
    proj = np.array(proj).reshape((X.shape[0],k))
    return proj

def plot_sectors(proj, sectors, sectors_to_plot):
    df = pd.concat([pd.DataFrame(proj), sectors], axis=1)
    df = df.loc[df['GICS Sector'].isin(sectors_to_plot)]
    fig, ax = plt.subplots()
    ax.margins(0.05)
    for name, group in df.groupby('GICS Sector'):
        ax.plot(group[0], group[1], marker='o', linestyle='', label=name)
    ax.legend()
    plt.show()


def load_shares():
    """ Q1 C """
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
    return symbols, pd.DataFrame(np.array(prices)), sectors


def q1():
    """ Q1 A """
    # df = pd.read_csv('prices.csv')
    # df.head(5)
    # # b TODO: explain in word
    # mask = df['date'].apply(lambda x: x[:4] == '2016')
    # df = df[mask]
    # df = df[df['symbol'] == 'AAPL'].reset_index()
    # apple_close_prices = df.close
    # apple_close_prices.plot()
    # plt.show()
    # c TODO: explain in word

    """ Q1 E """
    symbols, prices, sectors = load_shares()
    proj = pca_project(prices, 2)
    # plot_sectors(proj, sectors, ['Energy', 'Information Technology'])

    """F"""
    modifies_prices = (prices.apply(lambda x: ln_transformation(x))).T
    proj_modified = pca_project(modifies_prices, 2)
    # plot_sectors(proj_modified, sectors, ['Energy', 'Information Technology'])

    """ G """
    # plot_sectors(proj_modified, sectors, ['Energy', 'Information Technology','Real Estate'])

    """ H """
    proj_data = pd.DataFrame(proj_modified.T)
    # proj_data_special = proj_data.loc[proj_modified[0]<-56]
    # special_stock_symbol = symbols[proj_data_special.index[0]]
    plot_sectors(proj_modified, sectors, sectors)

    symbols_close_prices_data = pd.concat([pd.DataFrame(symbols), prices], axis=1)
    ZTS_close_prices = symbols_close_prices_data[symbols_close_prices_data["symbol"] == 'ZTS'].T[1:]
    ZTS_close_prices.plot()
    plt.legend("ZTS Stock")
    plt.show()


def ln_transformation(x):
    for i in range(len(x) - 1):
        x[i] = np.log(x[i + 1]) - np.log(x[i])
    return x


def main():
    q1()
    # X = np.array([[0, 0, 0],
    #               [3, 3, 5],
    #               [6, 8, 6],
    #               [9, 7, 9]])
    # proj = pca_project(X, 2)


if __name__ == '__main__':
    main()
