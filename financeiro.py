import pandas as pd
from pandas_datareader import data
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import requests


INDEXES_PATH = "http://bvmf.bmfbovespa.com.br/indices/"
INDEXES_PATH += "ResumoCarteiraTeorica.aspx?Indice=[INDEX]&idioma=pt-br"


class AIFinancial:
    def __init__(self, df, df_test):
        self.df = df
        self.df_test = df_test

    def heatmap_train(self, filters=[]):
        if len(filters) == 0:
            sns.heatmap(self.df.corr(), annot=True)
        else:
            sns.heatmap(self.df[filters].corr(), annot=True)

    def heatmap_test(self, filters=[]):
        if len(filters) == 0:
            sns.heatmap(self.df_test.corr(), annot=True)
        else:
            sns.heatmap(self.df_test[filters].corr(), annot=True)

    def kmeans_cluster(self, num):
        retornos = 100 * self.df[self.df.columns].pct_change()
        test_col = [
            each for each in self.df.columns if each in self.df_test.columns]
        retornos_test = self.df_test[test_col].pct_change()
        stocks = retornos.columns
        X = np.array([[np.std(retornos[sto]),
                           np.mean(retornos[sto])] for sto in stocks])
        X_test = np.array([[np.std(retornos_test[sto]),
                                  np.mean(retornos_test[sto])]
                                 for sto in test_col])
        N = num
        kmeans = KMeans(n_clusters=N, random_state=0).fit(X)
        y_kmeans = kmeans.predict(X)
        fig = plt.subplots(figsize=(20, 5))
        ax1 = plt.subplot(1, 1, 1)
        ax1.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
        ax1.set_title('Retornos de ' + str(len(stocks)) +
                      ' Ações do Índice Amplo', fontsize=20)
        ax1.set_xlabel('Risco [%]', fontsize=25)
        ax1.set_ylabel('Retorno [%]', fontsize=25)

        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=150, alpha=0.2)

        best = []
        for i in range(N):
            ind = retornos[retornos.columns[np.where(y_kmeans == i)[0]]].mean(
            )/retornos[retornos.columns[np.where(y_kmeans == i)[0]]].std()
            best.append(ind[ind == np.max(ind)])
        rb = list(pd.DataFrame(best).columns)
        print("Ativos com Melhor Relação em Cada Cluster:", rb)

        for r in rb:
            plt.text(
                X[stocks == r][0][0],
                X[stocks == r][0][1],
                r, fontsize=15)
        return fig, rb


class FinancialData:
    def __init__(self):
        pass

    def get_data_by_indexes(self, indexes):
        data = {}
        ativos = []
        for index in indexes:
            url = INDEXES_PATH.replace("[INDEX]", index)
            page = requests.get(url)
            df = pd.read_html(page.content)
            codigos = [x for x in df[0]["Código"][:-1]]
            ativos.extend(codigos)
            data[index] = codigos
        print("Número total de ativos: {}".format(len(list(set(ativos)))))
        for each in data.keys():
            print("INDEX: {}\t ATIVOS: {}".format(each, len(data[each])))
        return data

    def extract_data_from_actives(self, actives, event_date, source):
        list_of_df = []
        error_actives = []
        for active in tqdm(actives):
            try:
                df = data.DataReader(active,
                                     source,
                                     event_date[0],
                                     event_date[1])["Adj Close"]
                list_of_df.append(df)
            except Exception:
                error_actives.append(active)
                # print("Error on active {}".format(active))

        df = pd.concat(list_of_df, axis=1)
        df.columns = [each for each in actives
                      if each not in error_actives]
        return df

    def extract_volumes_from_actives(self, actives, event_date, source):
        list_of_df = []
        error_actives = []
        for active in tqdm(actives):
            try:
                df = data.DataReader(active,
                                     source,
                                     event_date[0],
                                     event_date[1])["Volume"]
                list_of_df.append(df)
            except Exception:
                error_actives.append(active)
                # print("Error on active {}".format(active))

        df = pd.concat(list_of_df, axis=1)
        df.columns = [each for each in actives
                      if each not in error_actives]
        return df

    def get_data_train_and_test(self, data, train_date, test_date):
        train = {}
        train_v = {}
        test = {}
        for each in data.keys():
            print("FUNDO: {}".format(each))
            actives = ["{}.SA".format(x) for x in data[each]]
            train_v[each] = self.extract_volumes_from_actives(actives,
                                                              train_date,
                                                              "yahoo")

            test[each] = self.extract_data_from_actives(actives,
                                                        test_date,
                                                        "yahoo")

            train[each] = self.extract_data_from_actives(actives,
                                                         train_date,
                                                         "yahoo")
        return train, test, train_v

    def plot_data_segmentation(self, df, column=None):
        if column is None:
            column = df.columns

        plt.figure(figsize=(40, 10))
        plot = sns.heatmap(df[column].isnull(), cbar=False, cmap="summer")
        return plot

    def cut_data_with_th(self, th, df):
        for each in df.columns:
            is_null_por = df[each].isnull().sum() / df[each].shape[0]
            if is_null_por > th:
                df.drop(each, axis='columns', inplace=True)
        return df

    def preprocessing_segmetation_data(self, df):
        print("Segmentacao de dados antes: ")
        _ = self.plot_data_segmentation(df)
        df = self.cut_data_with_th(0.3, df)
        df = df.dropna()
        _ = self.plot_data_segmentation(df)
        return df

    def get_sharpes(self, df):
        sharpes = {}
        df = 100 * df.diff()/df.iloc[0]
        for active in df.columns:
            sharpes[active] = ((df[active].mean())/df[active].std())

        return sharpes

    def get_max_sharpe(self, dict_sharpe, choices):
        best = ""
        maxx = -999999999
        for each in choices:
            if dict_sharpe[each] > maxx:
                maxx = dict_sharpe[each]
                best = each

        return best, maxx

    def get_best_actives(self, df, df_test, num_best, index):
        df = self.preprocessing_segmetation_data(df)
        df_test = self.preprocessing_segmetation_data(df_test)
        sharpes = self.get_sharpes(df)
        ai = AIFinancial(df, df_test)
        _, list_best = ai.kmeans_cluster(num_best)
        print(list_best)
        sorted_sharpe = sorted(sharpes.items(), key=lambda kv: kv[1])
        dict_sharpe = dict(sorted_sharpe)
        best, maxx = self.get_max_sharpe(dict_sharpe, list_best)
        result = (index, (best, maxx))
        return result


class FinancialSignal:
    def __init__(self, actives, event_dates):
        self.actives = actives
        self.event_dates = event_dates

    def extract_data_from_actives(self, source):
        list_of_df = []
        error_actives = []
        for active in tqdm(self.actives):
            try:
                df = data.DataReader(active,
                                     source,
                                     self.event_dates[0],
                                     self.event_dates[1])["Adj Close"]
                list_of_df.append(df)
            except Exception:
                error_actives.append(active)
                print("Error on active {}".format(active))

        df = pd.concat(list_of_df, axis=1)
        df.columns = [each for each in self.actives
                      if each not in error_actives]
        return df

    def plot_data_segmentation(self, df, column=None):
        if column is None:
            column = df.columns

        plt.figure(figsize=(40, 10))
        plot = sns.heatmap(df[column].isnull(), cbar=False, cmap="summer")
        return plot

    def plot_days_close(self, df, timestamp):
        plt.figure(figsize=(20, 14))
        return df[df.columns[:timestamp]].plot()

    def browniano_movement(self, active, po, time, mu, sigma):
        w = np.random.normal(0, 1, time)
        t = np.arange(0, time)
        p2 = po+mu*t+sigma*w
        fig = plt.subplots(figsize=(15, 5))
        plt.plot(p2)
        plt.ylabel('Preço [R$]', fontsize=15)
        plt.xlabel('tempo [s]', fontsize=15)
        plt.title('Preço Modelado com Movimento Browniano {}'.format(active),
                  fontsize=15)
        new_df = pd.DataFrame(p2, columns=[active])
        return fig, p2, new_df

    def browniano_geometric_movement(self, active, po, time, mu, sigma):
        w = np.random.normal(0, 1, time)
        t = np.arange(0, time)
        p2 = po*np.exp((mu-(sigma**2)/2)*t+sigma*w)

        fig = plt.subplots(figsize=(15, 5))
        plt.plot(p2)
        plt.ylabel('Preço [R$]', fontsize=15)
        plt.xlabel('tempo [s]', fontsize=15)
        plt.title('Preço Modelado com movimento browniano Geometrico',
                  fontsize=15)
        new_df = pd.DataFrame(p2, columns=[active])
        return fig, p2, new_df

    def active_returns(self, df, pt, active):
        fig = plt.subplots(figsize=(7, 4))
        r1m = (df[active][1:].values/df[active][0:-1].values)-1
        r1m[np.where(r1m >= 0)] = 0
        plt.bar(range(r1m.shape[0]), 100*r1m, color='red', edgecolor='black')

        r1m = (df[active][1:].values/df[active][0:-1].values)-1
        r1m[np.where(r1m < 0)] = 0
        plt.bar(range(r1m.shape[0]), 100*r1m, color='blue', edgecolor='black')

        plt.ylabel('Retorno [%]', fontsize=15)
        plt.xlabel('tempo [s]', fontsize=15)
        plt.title('Retorno Modelado com movimento browniano Geometrico',
                  fontsize=15)

        return fig

    def active_volatility(self, df, active):
        r1 = df[active][1:].values/df[active][0:-1].values-1
        df["r1"] = 100*np.concatenate(([0], r1), axis=0)
        fig = plt.subplots(figsize=(20, 5))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(df[active], color='m', label='Preço')
        ax1.plot(
            df[active].rolling(
                window=6).mean(),
            color='b',
            label='Média Móvel Preço')

        ax2 = ax1.twinx()
        ax2.plot(
            df['r1'].rolling(
                window=5).std(),
            color='g',
            label='Desvio Móvel Preço')
        ax1.set_title('Ativo 1', fontsize=25)
        ax1.set_ylabel('Preço [R$]', fontsize=15)
        ax2.set_ylabel('Desvio [R$]', fontsize=15)
        ax1.set_xlabel('Tempo [Dias]', fontsize=15)
        ax1.legend(loc='upper left', fontsize=15)
        ax2.legend(loc='lower left', fontsize=15)
        return fig
