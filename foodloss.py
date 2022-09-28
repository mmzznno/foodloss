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

print(train) 
#[207 rows x 13 columns]
print(test)
#[40 rows] y:non


#投稿用ファイル
submit = pd.read_csv("../input/sample_submission.csv", header=None)

#print(submit) 40 rows


#仮説1　気温・雨雲・相対湿度が売上に影響するのでは？

#仮説2　 社内イベント時は売上が多いのでは？

# 箱ひげ図で検証
sns.boxplot(x="y", y="event", data=train)

# x軸にラベルを付けて表示
plt.title("sales of each event content")
plt.xlabel("sales")
plt.show()

train["event"] = train.apply(lambda x: 1 if x["event"] == "ママの会" in x["event"] else 0, axis=1)
test["event"] = test.apply(lambda x: 1 if x["event"] == "ママの会"  in x["event"] else 0, axis=1)

#仮説3　 スペシャルメニュー時は売上が多いのでは？

# 箱ひげ図で検証
sns.boxplot(x="y", y="remarks", data=train)

# x軸にラベルを付けて表示
plt.title("sales of each remarks content")
plt.xlabel("sales")
plt.show()

train["remarks"] = train.apply(lambda x: 1 if x["remarks"] == "お楽しみメニュー" in x["remarks"] else 0, axis=1)
test["remarks"] = test.apply(lambda x: 1 if x["remarks"] == "お楽しみメニュー"  in x["remarks"] else 0, axis=1)


# カラム名のリスト
features =["remarks","event","temperature","cloud_amount","humidity"]

#学習データ（206）から分割
#学習データの説明変数、目的変数

train_X = train[features]
train_y = train["y"]

x_train, x_test, y_train, y_test = train_test_split(train_X, train_y, random_state=0)

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
print(pred2)

submit[1] = pred2
submit.to_csv("submit.csv", index=False, header=False)


#######################################################
# RMSEの計算
var = RMSE(y_test, pred1)
print(var)


# 評価データの販売数は、変数test_yに代入
# 販売数の予測値は、変数pred1に代入

# 折れ線グラフを描画

# 評価データの販売数でグラフを描く

#　評評価データの販売数が存在しないため以下省略
plt.plot(y_test.values, label="actual")

# 予測値でグラフを描く

plt.plot(pred1, label="forecast")

plt.title("sales of box lunch")
plt.xlabel("time step")
plt.ylabel("sales")
plt.legend()
plt.show()

######################################################