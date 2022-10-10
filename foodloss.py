#ライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

#線形回帰のインポート
from sklearn.linear_model import LinearRegression

# MSEのインポート
from sklearn.metrics import mean_squared_error

# 重回帰モデルの初期設定
lm = LinearRegression()

# RMSE関数を作る
def RMSE(var1, var2):
    
    # まずMSEを計算
    mse = mean_squared_error(var1,var2)
    
    # 平方根を取った値を返す
    return np.sqrt(mse)

# データ読み込み

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.tail(120)) 
#[207 rows x 13 columns]
#[40 rows] y:nan

#投稿用ファイル
submit = pd.read_csv("../input/sample_submission.csv", header=None)

#print(submit) 40 rows
#print("x") 
#print(train.head(11)) 

#仮説0　曜日と売り上げの関係？ 
#箱ひげ図で検証
#sns.boxplot(x="y", y="week", data=train)

df_week = train.loc[:,["week"]]
train_week = pd.get_dummies(df_week)

df_week = test.loc[:,["week"]]
test_week = pd.get_dummies(df_week)

#結合1
train = pd.concat([train, train_week], axis=1, sort =True)
test = pd.concat([test, test_week], axis=1, sort =True)

#print(train.head())
#[207 rows x 18 columns]
#print(test.shape)

#仮説1　気温・雨雲・相対湿度が売上に影響するのでは？

#気温ダミー変数
df_temperature = train.loc[:,["temperature"]]
dummy_df = df_temperature

print(df_temperature)

dummy_df["temperature[:12]"] = df_temperature.apply(lambda row: float(row.temperature < 15.6), axis=1)         
dummy_df["temperature[12:22]"] = df_temperature.apply(lambda row: float(row.temperature >= 15.6 and row.temperature < 22), axis=1)
dummy_df["temperature[23:]"] = df_temperature.apply(lambda row: float(row.temperature  >= 22), axis=1)

df_temperature_train_dummy = dummy_df.drop(["temperature"], axis=1)

#print(df_temperature_train_dummy)

#test
df_temperature = test.loc[:,["temperature"]]
dummy_df = df_temperature

dummy_df["temperature[:12]"] = df_temperature.apply(lambda row: float(row.temperature < 15.6), axis=1)         
dummy_df["temperature[12:22]"] = df_temperature.apply(lambda row: float(row.temperature >= 15.6 and row.temperature < 22), axis=1)

dummy_df["temperature[23:]"] = df_temperature.apply(lambda row: float(row.temperature  >= 22), axis=1)

df_temperature_test_dummy = dummy_df.drop(["temperature"], axis=1)

#print(df_temperature_test_dummy)


#結合2
train = pd.concat([train, df_temperature_train_dummy], axis=1, sort =True)
test = pd.concat([test, df_temperature_test_dummy], axis=1, sort =True)


#x=train["temperature"]
#y=train["y"]
#plt.scatter(x, y)
#plt.show()

#仮説1 イベントが売上に影響するのでは？
# 箱ひげ図で検証
sns.boxplot(x="y", y="event", data=train)

# x軸にラベルを付けて表示
plt.title("sales of each event content")
plt.xlabel("sales")
plt.show()

train["event"] = train.apply(lambda x: 2 if x["event"] == "ママの会" in x["event"] else 1, axis=1)
test["event"] = test.apply(lambda x: 2 if x["event"] == "ママの会"  in x["event"] else 1, axis=1)

train["event"] = train.apply(lambda x: 0 if x["event"] == "キャリアアップ支援セミナー" in x["event"] else 1, axis=1)
test["event"] = test.apply(lambda x: 0 if x["event"] == "キャリアアップ支援セミナー"  in x["event"] else 1, axis=1)

#仮説2　 肉類は売上が多いのでは？

train["kcal"] = train["kcal"].fillna(train["kcal"].median())
test["kcal"] = test["kcal"].fillna(test["kcal"].median())

train["kcal"] = train["kcal"].apply(lambda x: 1 if x> 408.5  else 0)
test["kcal"] = test["kcal"].apply(lambda x: 1 if x> 408.5  else 0)
#print(train["kcal"])

#仮説3 　給料日は外食？

train["payday"] = train["payday"].fillna(0)
test["payday"] = test["payday"].fillna(0)
#print(train["payday"])

#仮説4 　雨天は外食？
train["precipitation"] = train.apply(lambda x: 1 if x["precipitation"] == "--" in x["precipitation"] else 0, axis=1)
test["precipitation"] = test.apply(lambda x: 1 if x["precipitation"] == "--"  in x["precipitation"] else 0, axis=1)

#print(train["precipitation"])


#仮説5 完売プラグの影響は？
# 箱ひげ図で検証
#sns.boxplot(x="soldout", y="y", data=train)

# x軸にラベルを付けて表示
#plt.title("sales of 売り切れ")
#plt.xlabel("sales")
#plt.show()

#仮説6 雨雲・相対湿度が売上に影響するのでは？

#雲量ダミー変数
df_cloud_amount = train.loc[:,["cloud_amount"]]
dummy_df = df_cloud_amount

#print(df_cloud_amount)

dummy_df["cloud_amount[:4]"] = df_cloud_amount.apply(lambda row: int(row.cloud_amount < 4), axis=1)         
dummy_df["cloud_amount[4:7]"] = df_cloud_amount.apply(lambda row: int(row.cloud_amount >= 4 and row.cloud_amount < 7), axis=1)

dummy_df["cloud_amount[7:]"] = df_cloud_amount.apply(lambda row: int(row.cloud_amount  >= 7), axis=1)

df_cloud_amount_train_dummy = dummy_df.drop(["cloud_amount"], axis=1)

#print(df_cloud_amount_train_dummy)

#test
df_cloud_amount = test.loc[:,["cloud_amount"]]
dummy_df = df_cloud_amount

dummy_df["cloud_amount[:4]"] = df_cloud_amount.apply(lambda row: int(row.cloud_amount < 4), axis=1)         
dummy_df["cloud_amount[4:7]"] = df_cloud_amount.apply(lambda row: int(row.cloud_amount >= 4 and row.cloud_amount < 7), axis=1)

dummy_df["cloud_amount[7:]"] = df_cloud_amount.apply(lambda row: int(row.cloud_amount  >= 7), axis=1)

df_cloud_amount_test_dummy = dummy_df.drop(["cloud_amount"], axis=1)

#print(df_cloud_amount_test_dummy)


#結合3
train = pd.concat([train, df_cloud_amount_train_dummy], axis=1, sort =True)
test = pd.concat([test, df_cloud_amount_test_dummy], axis=1, sort =True)


train["remarks"] = train.apply(lambda x: 1 if x["remarks"] == "お楽しみメニュー" in x["remarks"] else 0, axis=1)
test["remarks"] = test.apply(lambda x: 1 if x["remarks"] == "お楽しみメニュー"  in x["remarks"] else 0, axis=1)

#カラム名のリスト
#features =["week_月","week_火","week_水","week_木","week_金","precipitation","remarks","event","temperature[:12]","temperature[12:22]","temperature[23:]","cloud_amount[:4]","cloud_amount[4:7]","cloud_amount[7:]"]
#features =["precipitation","remarks","event","temperature[:12]","temperature[12:22]","temperature[23:]","cloud_amount[:4]","cloud_amount[4:7]","cloud_amount[7:]"]
features =["week_月","week_火","week_水","week_木","week_金","kcal","precipitation","remarks","event","temperature[:12]","temperature[12:22]","temperature[23:]","cloud_amount[:4]","cloud_amount[4:7]","cloud_amount[7:]","payday"]


#学習データ（206）から分割
#学習データの説明変数、目的変数

train_X = train[features]
train_y = train["y"]

#データの後半に絞ってみる。
train_X = train_X[126:]
train_y = train_y[126:]

#print("test = train_X[126:]")
#print(train_X.head(50))
#print(train_y.head(50))


x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, random_state=0)

#train =data[data["flag"] == 1 ]
#test = data[data["flag"] == 0]

# モデルの学習
X = x_train
y = y_train
lm = LinearRegression()
lm.fit(X,y)

# 学習済みモデルlmの回帰係数を、DataFrameにして表示
# 説明変数の名称は変数featuresにリストで代入
df = pd.DataFrame(lm.coef_, index=features, columns=["回帰係数"])
print(df)

# 予測値を算出
lm = LinearRegression()
lm.fit(x_test,y_test)
pred1 = lm.predict(x_test)

# 提出用予測値(データ40）を算出
test_X = test[features]
pred2 = lm.predict(test_X)
#print(pred2)

submit[1] = pred2
submit.to_csv("submit.csv", index=False, header=False)

#######################################################
# RMSEの計算
var = RMSE(y_test, pred1)
print(var)

# 評価データの販売数は、変数test_yに代入されています。
# 販売数の予測値は、変数pred1に代入

# 折れ線グラフを描画# 評価データの販売数でグラフを描く
plt.plot(y_test.values, label="actual")
# 予測値でグラフを描く
plt.plot(pred1, label="forecast")
plt.title("sales of box lunch")
plt.xlabel("time step")
plt.ylabel("sales")
plt.legend()
plt.show()

#######################################################