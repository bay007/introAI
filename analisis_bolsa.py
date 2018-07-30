import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def get_date(filename):
    dates = []
    prices = []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            dates.append(int(row[0].split("/")[0]))
            prices.append(float(row[3]))
    return (dates, prices)


def predict_prices(dates, prices, x):

    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1))
    svr_lin = SVR(kernel='linear', C=1e2)
    #svr_poly = SVR(kernel='poly', C=1e1, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.1)

    svr_lin.fit(dates, prices)
    #svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='red', label='linear')
    #plt.plot(dates, svr_poly.predict(dates), color='green', label='polinomial')
    plt.plot(dates, svr_rbf.predict(dates), color='blue', label='RBF')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title("Support Vector Regression")
    plt.legend()
    plt.show()

    # , svr_poly.predict(x)[0]
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0]


if __name__ == '__main__':
    dates, prices = get_date("HistoricalQuotes.csv")
    print(predict_prices(dates, prices, 2019))
