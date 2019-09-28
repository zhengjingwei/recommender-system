# -*-coding:utf-8-*-
"""
    GBDT模型 电信客户流失预测
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

class ChurnPredWithGBDT:
    def __init__(self):
        self.file = "../data/telecom-churn/telecom-churn-prediction-data.csv"
        self.data = self.feature_transform()
        self.train, self.test = self.split_data()

    # 空缺值以0值填充
    def isNone(self, value):
        if value == " " or value is None:
            return "0.0"
        else:
            return value

    # 特征转换
    def feature_transform(self):
        if not os.path.exists("data/new-churn.csv"):
            print("Start Feature Transform ...")
            # 定义特征转换字典
            feature_dict = {
                "gender": {"Male": "1", "Female": "0"},
                "Partner": {"Yes": "1", "No": "0"},
                "Dependents": {"Yes": "1", "No": "0"},
                "PhoneService": {"Yes": "1", "No": "0"},
                "MultipleLines": {"Yes": "1", "No": "0", "No phone service": "2"},
                "InternetService": {"DSL": "1", "Fiber optic": "2", "No": "0"},
                "OnlineSecurity": {"Yes": "1", "No": "0", "No internet service": "2"},
                "OnlineBackup": {"Yes": "1", "No": "0", "No internet service": "2"},
                "DeviceProtection": {"Yes": "1", "No": "0", "No internet service": "2"},
                "TechSupport": {"Yes": "1", "No": "0", "No internet service": "2"},
                "StreamingTV": {"Yes": "1", "No": "0", "No internet service": "2"},
                "StreamingMovies": {"Yes": "1", "No": "0", "No internet service": "2"},
                "Contract": {"Month-to-month": "0", "One year": "1", "Two year": "2"},
                "PaperlessBilling": {"Yes": "1", "No": "0"},
                "PaymentMethod": {
                    "Electronic check": "0",
                    "Mailed check": "1",
                    "Bank transfer (automatic)": "2",
                    "Credit card (automatic)": "3",
                },
                "Churn": {"Yes": "1", "No": "0"},
            }
            fw = open("data/new_churn.csv", "w")
            fw.write(
                "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
                "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
                "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\n"
            )
            for line in open(self.file, "r").readlines():
                if line.startswith("customerID"):
                    continue
                customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, \
                InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, \
                StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn \
                    = line.strip().split(",")
                _list = list()
                _list.append(customerID)
                _list.append(self.isNone(feature_dict["gender"][gender]))
                _list.append(self.isNone(SeniorCitizen))
                _list.append(self.isNone(feature_dict["Partner"][Partner]))
                _list.append(self.isNone(feature_dict["Dependents"][Dependents]))
                _list.append(self.isNone(tenure))
                _list.append(self.isNone(feature_dict["PhoneService"][PhoneService]))
                _list.append(self.isNone(feature_dict["MultipleLines"][MultipleLines]))
                _list.append(
                    self.isNone(feature_dict["InternetService"][InternetService])
                )
                _list.append(
                    self.isNone(feature_dict["OnlineSecurity"][OnlineSecurity])
                )
                _list.append(self.isNone(feature_dict["OnlineBackup"][OnlineBackup]))
                _list.append(
                    self.isNone(feature_dict["DeviceProtection"][DeviceProtection])
                )
                _list.append(self.isNone(feature_dict["TechSupport"][TechSupport]))
                _list.append(self.isNone(feature_dict["StreamingTV"][StreamingTV]))
                _list.append(
                    self.isNone(feature_dict["StreamingMovies"][StreamingMovies])
                )
                _list.append(self.isNone(feature_dict["Contract"][Contract]))
                _list.append(
                    self.isNone(feature_dict["PaperlessBilling"][PaperlessBilling])
                )
                _list.append(self.isNone(feature_dict["PaymentMethod"][PaymentMethod]))
                _list.append(self.isNone(MonthlyCharges))
                _list.append(self.isNone(TotalCharges))
                _list.append(feature_dict["Churn"][Churn])
                fw.write(",".join(_list))
                fw.write("\n")
            return pd.read_csv("data/new_churn.csv")
        else:
            return pd.read_csv("data/new_churn.csv")

    # 数据集拆分为训练集和测试集
    def split_data(self):
        train, test = train_test_split(
            self.data,
            test_size=0.1,
            random_state=40
        )
        return train, test

    # 调用skleran进行模型训练
    def train_model(self):
        print("Start Train Model ... ")
        lable = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.train.columns if x not in [lable, ID]]
        x_train = self.data[x_columns]
        y_train = self.data[lable]
        gbdt = GradientBoostingClassifier(
            learning_rate=0.1, n_estimators=100, max_depth=7
        )
        gbdt.fit(x_train, y_train)
        return gbdt

    # 模型评估
    def evaluate(self, gbdt):
        lable = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.test.columns if x not in [lable, ID]]
        x_test = self.data[x_columns]
        y_test = self.data[lable]
        y_pred = gbdt.predict_proba(x_test)
        new_y_pred = list()
        for y in y_pred:
            # y[0] 表示样本label=0的概率 y[1]表示样本label=1的概率
            new_y_pred.append(1 if y[1] > 0.5 else 0)
        mse = mean_squared_error(y_test, new_y_pred)
        print("MSE: %.4f" % mse)
        accuracy = metrics.accuracy_score(y_test.values, new_y_pred)
        print("Accuracy : %.4g" % accuracy)
        auc = metrics.roc_auc_score(y_test.values, new_y_pred)
        print("AUC Score : %.4g" % auc)

if __name__ == "__main__":
    pred = ChurnPredWithGBDT()
    gbdt = pred.train_model()
    pred.evaluate(gbdt)
