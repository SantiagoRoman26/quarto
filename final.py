# MIDTERM
## Importar librerias


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import RocCurveDisplay
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 42

import nltk
from nltk.corpus import stopwords
sns.set(style="whitegrid")
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

## Limpieza de los datos

url = "https://raw.githubusercontent.com/erickedu85/dataset/refs/heads/master/tweets/1500_tweets_con_toxicity.csv"
df = pd.read_csv(url)
df.info()
df.head()




if 'tweetId' in df.columns:
    df.drop_duplicates(subset='tweetId', inplace=True)

# Revisar nulos por columna
print("Nulos por columna:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

### Eliminar valores nulos o vacios en la columna target


df = df.dropna(subset=['toxicity_score'])
df.isnull().sum().sort_values(ascending=False)

### Seleccionar columnas relevantes

columns_to_keep = [
    'content', 'isReply', 'authorVerified', 'has_profile_picture',
    'authorFollowers', 'account_age_days', 'mentions_count',
    'hashtags_count', 'content_length', 'sentiment_polarity',
    'toxicity_score'
]
df = df[columns_to_keep]

### Limpieza del texto


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)         # URLs
    text = re.sub(r"@\w+", "", text)            # menciones
    text = re.sub(r"#\w+", "", text)            # hashtags
    text = re.sub(r"[^a-záéíóúñü\s]", "", text) # puntuación
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['clean_content'] = df['content'].apply(clean_text)



## EDA
### Distribucion 

# Estadísticas numéricas
print(df.describe().T)

plt.figure(figsize=(8,4))
sns.histplot(df['toxicity_score'], bins=30, kde=True)
plt.title('Distribución de TOXICITY_SCORE')
plt.xlabel('Toxicity score (0-1)')
plt.show()


### Boxplot para ver outliers


plt.figure(figsize=(6,3))
sns.boxplot(x=df['toxicity_score'])
plt.title('Boxplot TOXICITY_SCORE')
plt.show()


### Conteos de target



if 'isReply' in df.columns:
    plt.figure(figsize=(6,3))
    sns.countplot(x='isReply', data=df)
    plt.title('isReply distribución')
    plt.show()

# Mostrar 3 hallazgos clave que posiblemente verás
print("Sugerencia de hallazgos: (1) Distribución sesgada hacia valores bajos de toxicidad; (2) muchos registros con pocos followers; (3) presencia de valores nulos en metadatos si no fueron limpiados.")


### Umbral de Toxicidad




threshold = 0.6
df['target_toxic'] = (df['toxicity_score'] >= threshold).astype(int)

# Ver la proporción de clases
print("Distribución binaria (0=no tóxico, 1=tóxico):")
print(df['target_toxic'].value_counts(normalize=True))
sns.countplot(x='target_toxic', data=df)
plt.title(f'Distribución target con umbral = {threshold}')
plt.show()



## Preprocesamiento y codificación:
### Preparar features a usar

text_feature = ['clean_content']
categorical_features = ['isReply', 'authorVerified', 'has_profile_picture']
numeric_features = [
    'authorFollowers', 'account_age_days', 'mentions_count',
    'hashtags_count', 'content_length', 'sentiment_polarity'
]

### Generar los Transformers para los modelos

tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
cat_encoder = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('text', tfidf, text_feature),
        ('cat', cat_encoder, categorical_features),
        ('num', scaler, numeric_features)
    ],
    remainder='drop',  # otras columnas se eliminan
    sparse_threshold=0  # forzar salida densa si es necesario (sklearn >=1.2)
)

### División train/test


X = df[text_feature + categorical_features + numeric_features]
y_class = df['target_toxic']        # para clasificación binaria
y_reg = df['toxicity_score']        # para regresión (continuo)

X_train, X_test, y_train_cl, y_test_cl = train_test_split(
    X, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
)

# Para regresión (mismo split de X, pero con y_reg alineado):
# usamos el mismo índice de train/test para evitar fugas:
_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
)


## Clasificación
### Pipeline para el Logistic Regression y entrenamiento


clf_logistic = Pipeline([
    ('preproc', preprocessor),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))
])

# Entrenar
clf_logistic.fit(X_train, y_train_cl)




### Predicciones


y_pred_log = clf_logistic.predict(X_test)
y_proba_log = clf_logistic.predict_proba(X_test)[:,1]

print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test_cl, y_pred_log))
print("Precision:", precision_score(y_test_cl, y_pred_log, zero_division=0))
print("Recall:", recall_score(y_test_cl, y_pred_log, zero_division=0))
print("F1:", f1_score(y_test_cl, y_pred_log, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test_cl, y_proba_log))


### Matriz de confusión


cm = confusion_matrix(y_test_cl, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no_tox','tox'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Logistic')
plt.show()


### Curva ROC



RocCurveDisplay.from_estimator(clf_logistic, X_test, y_test_cl)
plt.title('ROC Curve - Logistic')
plt.show()


### 


## Regresión
### Pipeline y entrenamiento


reg_lin = Pipeline([
    ('preproc', preprocessor),
    ('linreg', LinearRegression())
])

reg_rf = Pipeline([
    ('preproc', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE))
])

# Entrenar
reg_lin.fit(X_train, y_train_reg)
reg_rf.fit(X_train, y_train_reg)

### Predicciones


y_pred_lin = reg_lin.predict(X_test)
y_pred_rf = reg_rf.predict(X_test)

# Métricas
def regression_metrics(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

regression_metrics(y_test_reg, y_pred_lin, "LinearRegression")

### Scatter plots


plt.figure(figsize=(6,6))
plt.scatter(y_test_reg, y_pred_rf, alpha=0.5)
plt.plot([0,1],[0,1], 'r--')  # línea ideal
plt.xlabel("Toxicity real")
plt.ylabel("Toxicity predicho")
plt.title("Real vs Predicho - RandomForestRegressor")
plt.show()



### Histograma de errores


resid = y_test_reg - y_pred_rf
plt.figure(figsize=(6,4))
sns.histplot(resid, bins=30, kde=True)
plt.title("Errores residuales - RandomForestRegressor")
plt.xlabel("Residual (real - predicho)")
plt.show()


## Clustering
### Preparar features densos


text_pipeline_for_clust = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1500, ngram_range=(1,2))),
    ('svd', TruncatedSVD(n_components=50, random_state=RANDOM_STATE))
])

preprocessor_clust = ColumnTransformer(
    transformers=[
        ('text_svd', text_pipeline_for_clust, text_feature),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='drop'
)

# Obtener representación densa para clustering
X_clust = preprocessor_clust.fit_transform(X)  # usa todo el dataset para clustering
print("Forma de la matriz de features para clustering:", X_clust.shape)

### Elegir k con silhouette y aplicar KMeans


sil_scores = {}
for k in range(2,7):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_clust)
    sil = silhouette_score(X_clust, labels)
    sil_scores[k] = sil
    print(f"k={k}, silhouette={sil:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print("Mejor k por silhouette:", best_k)

### Ajustar KMeans


kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X_clust)
agg_final = AgglomerativeClustering(n_clusters=best_k).fit(X_clust)

# Añadir etiquetas de cluster al dataframe original para análisis
df_clustering = df.copy().reset_index(drop=True)
df_clustering['cluster_kmeans'] = kmeans_final.labels_
df_clustering['cluster_agg'] = agg_final.labels_

### Visualizar clusters


pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca.fit_transform(X_clust)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=df_clustering['cluster_kmeans'], palette='tab10', s=30)
plt.title('KMeans clusters (PCA 2D)')

plt.subplot(1,2,2)
sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=df_clustering['target_toxic'], palette='coolwarm', s=30)
plt.title('Target binario (PCA 2D)')
plt.show()

### Tabla de contingencia entre clusters y target


ct = pd.crosstab(df_clustering['cluster_kmeans'], df_clustering['target_toxic'], normalize='index')
print("Proporción por cluster vs target (KMeans):")

## Conclusiones


print("\n--- Resumen rápido ---")
print("Tamaño dataset tras limpiar toxicity_score:", df.shape[0])
print("Umbral usado para binarizar toxicity:", threshold)
print("Mejor k clustering por silhouette:", best_k)
