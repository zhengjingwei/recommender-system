# -*-coding:utf-8-*-
"""
    Author: Thinkgamer
    Desc:
        代码8-3： 第八章 LR模型 电信客户流失预测
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import pandas as pd

class ChurnPredWithLR:
    def __init__(self):
        self.file = "data/new_churn.csv"
        self.data = self.load_data()
        self.train, self.test = self.split()

    # 加载数据
    def load_data(self):
        return pd.read_csv(self.file)

    # 拆分数据集
    def split(self):
        train, test = train_test_split(
            self.data,
            test_size=0.1,
            random_state=40
        )
        return train, test

    # 模型训练
    def train_model(self):
        print("Start Train Model ... ")
        lable = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.train.columns if x not in [lable, ID]]
        x_train = self.data[x_columns]
        y_train = self.data[lable]
        # 定义模型
        lr = LogisticRegression(penalty="l2", tol=1e-4, fit_intercept=True) # tol停止求解的差值阈值
        lr.fit(x_train, y_train)
        return lr

    # 模型评估
    def evaluate(self, lr, type):
        lable = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.test.columns if x not in [lable, ID]]
        x_test = self.data[x_columns]
        y_test = self.data[lable]
        if type == 1:
            y_pred = lr.predict(x_test)
            new_y_pred = y_pred
        elif type == 2:
            y_pred = lr.predict_proba(x_test)
            new_y_pred = list()
            for y in y_pred:
                new_y_pred.append(1 if y[1] > 0.5 else 0)
        mse = mean_squared_error(y_test, new_y_pred)
        print("MSE: %.4f" % mse)
        accuracy = metrics.accuracy_score(y_test.values, new_y_pred)
        print("Accuracy : %.4g" % accuracy)
        auc = metrics.roc_auc_score(y_test.values, new_y_pred)
        print("AUC Score : %.4g" % auc)


if __name__ == "__main__":
    pred = ChurnPredWithLR()
    lr = pred.train_model()
    # type=1：表示输出0 、1 type=2：表示输出概率
    pred.evaluate(lr, type=1)
