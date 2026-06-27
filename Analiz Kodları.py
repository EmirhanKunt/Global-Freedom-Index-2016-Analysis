import numpy as np
import pandas as pd
import matplotlib.pyplot as mt
import seaborn as sb
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Görsel Ayarlar
mt.rcParams['figure.figsize'] = (10, 6)
sb.set_style("whitegrid")

# ==========================================
# 1. VERİ HAZIRLIĞI
# ==========================================
data = pd.read_csv(r"C:\Users\Emirhan Kunt\Desktop\Programlama\Python Klasörü\data\freedom.csv")
data_2016 = data[data["year"] == 2016].copy()

data_final = pd.get_dummies(data_2016, columns=['Status', 'Region_Name'], drop_first=True)
y = data_final["CL"]
X = data_final.drop(["CL", "country", "year"], axis=1).fillna(data_final.mean(numeric_only=True))

x_egitim, x_test, y_egitim, y_test = train_test_split(X, y, test_size=0.25, random_state=34)

scl = StandardScaler()
x_egitim_scl = scl.fit_transform(x_egitim)
x_test_scl = scl.transform(x_test)

model_isimleri, r2_skorlari, mape_skorlari = [], [], []

def skor_kaydet(isim, y_gercek, y_tahmin):
    model_isimleri.append(isim)
    r2_skorlari.append(r2_score(y_gercek, y_tahmin))
    mape_skorlari.append(mean_absolute_percentage_error(y_gercek, y_tahmin) * 100)

# ==========================================
# 2. DOĞRUSAL VE BOYUT İNDİRGEME MODELLERİ
# ==========================================

# --- 1. OLS (Linear Regression) ---
ols_model = LinearRegression().fit(x_egitim_scl, y_egitim)
ols_pred = ols_model.predict(x_test_scl)
skor_kaydet("OLS", y_test, ols_pred)
# [GÖRSEL 1]
mt.figure()
mt.scatter(y_test, ols_pred, alpha=0.6, color='blue')
mt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
mt.title("OLS: Gerçek vs Tahmin (Regresyon Çizgisi)")
mt.show()

# --- 2. PCR (Principal Component Regression) ---
from sklearn.model_selection import cross_val_score, KFold

# 1. En iyi bileşen sayısını bulmak için MSE hesaplama döngüsü
mse_listesi = []
cv = KFold(n_splits=10, shuffle=True, random_state=34)

for i in range(1, X.shape[1] + 1):
    pca = PCA(n_components=i)
    X_reduced = pca.fit_transform(x_egitim_scl)
    score = -1 * cross_val_score(LinearRegression(), X_reduced, y_egitim, cv=cv, scoring='neg_mean_squared_error').mean()
    mse_listesi.append(score)

# 2. Grafikteki en düşük noktayı (6. bileşen) baz alarak nihai modeli kurma
pca_final = PCA(n_components=6) # Grafikteki çakılma noktası 6 olarak belirlendi
X_pcr_egitim = pca_final.fit_transform(x_egitim_scl)
pcr_model = LinearRegression().fit(X_pcr_egitim, y_egitim)

# 3. Test seti tahmini ve skor kaydı
X_pcr_test = pca_final.transform(x_test_scl)
pcr_pred = pcr_model.predict(X_pcr_test)
skor_kaydet("PCR (n=6)", y_test, pcr_pred)

mt.figure()
mt.plot(np.arange(1, X.shape[1] + 1), mse_pcr, '-v', color='orange')
mt.title("PCR: Bileşen Sayısına Göre Hata (MSE)")
mt.show()

# --- 3. PLS (Partial Least Squares) ---
pls_model = PLSRegression(n_components=2).fit(x_egitim_scl, y_egitim)
pls_pred = pls_model.predict(x_test_scl).flatten()
skor_kaydet("PLS", y_test, pls_pred)
# [GÖRSEL 3] Bileşenlerin Hedefe Göre Dağılımı
X_pls = pls_model.transform(x_egitim_scl)
mt.figure()
mt.scatter(X_pls[:, 0], X_pls[:, 1], c=y_egitim, cmap='plasma')
mt.title("PLS: Bileşen 1 vs Bileşen 2 (Renkler CL Puanı)")
mt.show()

# --- 4. Ridge ---
ridge_cv = RidgeCV(alphas=np.logspace(-6, 6, 13)).fit(x_egitim_scl, y_egitim)
ridge_model = Ridge(alpha=ridge_cv.alpha_).fit(x_egitim_scl, y_egitim)
skor_kaydet("Ridge", y_test, ridge_model.predict(x_test_scl))
# [GÖRSEL 4] Katsayı Önem Barı
mt.figure()
pd.Series(ridge_model.coef_, index=X.columns).sort_values().plot(kind='barh', color='cyan')
mt.title("Ridge: Değişken Katsayılarının Sönümlenmesi")
mt.show()

# --- 5. Lasso ---
lasso_cv = LassoCV(cv=10, random_state=34).fit(x_egitim_scl, y_egitim)
lasso_model = Lasso(alpha=lasso_cv.alpha_).fit(x_egitim_scl, y_egitim)
skor_kaydet("Lasso", y_test, lasso_model.predict(x_test_scl))
# [GÖRSEL 5] Elenen Değişkenler (Sıfır Olanlar)
mt.figure()
pd.Series(lasso_model.coef_, index=X.columns).sort_values().plot(kind='barh', color='red')
mt.title("Lasso: Katsayılar (Sıfır Olanlar Elendi)")
mt.show()

# --- 6. ElasticNet ---
enet_cv = ElasticNetCV(cv=10, random_state=34).fit(x_egitim_scl, y_egitim)
enet_model = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_).fit(x_egitim_scl, y_egitim)
skor_kaydet("ElasticNet", y_test, enet_model.predict(x_test_scl))
# [GÖRSEL 6] Hata Yoğunluk Grafiği
mt.figure()
sb.kdeplot(y_test - enet_model.predict(x_test_scl), fill=True, color='purple')
mt.title("ElasticNet: Hata (Artık) Dağılımı")
mt.show()

# ==========================================
# 3. KOMŞULUK VE MESAFE TEMELLİ 
# ==========================================

# --- 7. KNN (K-Nearest Neighbors) ---
knn_model = KNeighborsRegressor(n_neighbors=5).fit(x_egitim_scl, y_egitim)
skor_kaydet("KNN", y_test, knn_model.predict(x_test_scl))
# [GÖRSEL 7] Komşu Sayısı Analizi
k_scores = []
for k in range(1, 21):
    k_scores.append(-cross_val_score(KNeighborsRegressor(n_neighbors=k),
                                     x_egitim_scl, y_egitim, cv=5, scoring='neg_mean_squared_error').mean())
mt.figure()
mt.plot(range(1, 21), k_scores, 'g-o')
mt.title("KNN: Komşu Sayısı (K) vs Hata")
mt.show()

# --- 8. SVR (Support Vector Regression) ---
svr_model = SVR(kernel="rbf", C=10, epsilon=0.1).fit(x_egitim_scl, y_egitim)
skor_kaydet("SVR", y_test, svr_model.predict(x_test_scl))
# [GÖRSEL 8] Hata Saçılımı (Residual Plot)
mt.figure()
mt.scatter(svr_model.predict(x_test_scl), y_test - svr_model.predict(x_test_scl), color='brown')
mt.axhline(0, color='black', linestyle='--')
mt.title("SVR: Tahminler vs Hatalar")
mt.show()

# ==========================================
# 4. AĞAÇ VE TOPLULUK MODELLERİ 
# ==========================================

# --- 9. CART (Decision Tree) ---
cart_model = DecisionTreeRegressor(max_depth=4).fit(x_egitim, y_egitim)
skor_kaydet("CART", y_test, cart_model.predict(x_test))
# [GÖRSEL 9] Karar Ağacı Şeması
mt.figure(figsize=(15,7))
plot_tree(cart_model, feature_names=X.columns, filled=True, fontsize=8)
mt.title("CART: Karar Ağacı Dallanma Şeması")
mt.show()

# --- 10. Bagging ---
bag_model = BaggingRegressor(n_estimators=100, random_state=34).fit(x_egitim, y_egitim)
skor_kaydet("Bagging", y_test, bag_model.predict(x_test))
# [GÖRSEL 10] Gerçek vs Tahmin Saçılımı
mt.figure()
mt.scatter(y_test, bag_model.predict(x_test), color='navy', alpha=0.5)
mt.title("Bagging: Tahmin Başarısı")
mt.show()

# --- 11. Random Forest ---
rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=34).fit(x_egitim, y_egitim)
skor_kaydet("Random Forest", y_test, rf_model.predict(x_test))
# [GÖRSEL 11] Feature Importance (Değişken Önemi)
mt.figure()
pd.Series(rf_model.feature_importances_, index=X.columns).sort_values().plot(kind='barh', color='darkgreen')
mt.title("Random Forest: Özgürlüğü Belirleyen En Önemli Faktörler")
mt.show()

# --- 12. GBM (Gradient Boosting Machine) ---
gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1).fit(x_egitim, y_egitim)
skor_kaydet("GBM", y_test, gbm_model.predict(x_test))
# [GÖRSEL 12] Hata Azalma Grafiği (Deviance)
mt.figure()
mt.plot(gbm_model.train_score_, label='Eğitim Hatası', color='darkred')
mt.title("GBM: Her Adımda Hatanın İyileşmesi")
mt.show()

# ==========================================
# 5. YAPAY SİNİR AĞLARI ( ANN)
# ==========================================

# --- 13. ANN (Multi-Layer Perceptron) ---
ann_model = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=34).fit(x_egitim_scl, y_egitim)
skor_kaydet("ANN", y_test, ann_model.predict(x_test_scl))
# [GÖRSEL 13] Loss Curve (Öğrenme Eğrisi)
mt.figure()
mt.plot(ann_model.loss_curve_, color='black')
mt.title("ANN: Yapay Sinir Ağının Öğrenme Süreci (Loss Curve)")
mt.show()

# ==========================================
# 6. SONUÇ TABLOSU
# ==========================================
final_df = pd.DataFrame({"Model": model_isimleri, "R2": r2_skorlari, "MAPE (%)": mape_skorlari}).sort_values(by="MAPE (%)")
print("\n🏆 KARŞILAŞTIRMA")
print(final_df)

mt.figure()
sb.barplot(x="MAPE (%)", y="Model", data=final_df, palette="viridis")
mt.title("Tüm Modellerin MAPE Karşılaştırması")
mt.show()

# Mevcut sonuç tablon üzerinden R2 kıyaslaması
final_df_r2 = final_df.sort_values(by="R2", ascending=False) # En yüksek başarı en üstte

mt.figure(figsize=(12, 7))
sb.barplot(x="R2", y="Model", data=final_df_r2, palette="plasma")
mt.title("Modellerin Belirlenme Katsayısı (R2) Kıyaslaması")
mt.xlabel("R2 Skoru (1.00'a Yakın Olması İdealdir)")
mt.axvline(0.70, color='red', linestyle='--', label='Kabul Edilebilir Eşik')
mt.legend()
mt.show()
