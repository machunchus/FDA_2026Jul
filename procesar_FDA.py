import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def procesar_muestras_fda(df_tidy, window_length=21, polyorder=2):
  group_cols = [
      'Dia_Cultivo',
      'Corrida',
      'Posicion',
      'Maceta_ID',
      'Condicion',
      'Masa',
  ]
  records = []

  for keys, group in df_tidy.groupby(group_cols):
    group = group.sort_values('Tiempo').reset_index(drop=True)
    t, g_net, a_channel = (
        group['Tiempo'].values,
        group['G_G0'].values,
        group['A'].values,
    )
    masa, dt = keys[5], np.median(np.diff(t))

    w_len = (
        window_length
        if len(t) >= window_length
        else (len(t) - 1 if len(t) % 2 == 0 else len(t))
    )
    dG_dt = savgol_filter(
        g_net, window_length=w_len, polyorder=polyorder, deriv=1, delta=dt
    )
    d2G_dt2 = savgol_filter(
        g_net, window_length=w_len, polyorder=polyorder, deriv=2, delta=dt
    )

    mask_sed = t >= 12.0
    idx_vmax = np.argmax(dG_dt[mask_sed])
    vmax = dG_dt[mask_sed][idx_vmax]
    t_vmax = t[mask_sed][idx_vmax]

    auc = np.trapz(g_net[t >= 15.0], t[t >= 15.0])
    idx_12 = np.argmin(np.abs(t - 12.0))
    delta_A = a_channel[idx_12] - a_channel[-1]

    records.append({
        'Dia_Cultivo': keys[0],
        'Corrida': keys[1],
        'Posicion': keys[2],
        'Maceta_ID': keys[3],
        'Condicion': keys[4],
        'Masa': masa,
        'Vmax_norm': vmax / masa,
        't_Vmax': t_vmax,
        'AUC_norm': auc / masa,
        'Delta_A': delta_A,
        'Int_Final_norm': g_net[-1] / masa,
        'Concavity': np.mean(d2G_dt2[t >= t_vmax]),
    })

  return pd.DataFrame(records)


def graficar_biplot_pca(df_features, features_cols):
  X = df_features[features_cols]
  X_scaled = StandardScaler().fit_transform(X)

  pca = PCA(n_components=2)
  scores = pca.fit_transform(X_scaled)
  loadings = pca.components_.T

  plt.figure(figsize=(9, 6))
  condiciones = df_features['Condicion'].unique()

  for cond in condiciones:
    mask = df_features['Condicion'] == cond
    plt.scatter(
        scores[mask, 0], scores[mask, 1], label=f'Condición {cond}', alpha=0.7
    )

  for i, feature in enumerate(features_cols):
    plt.arrow(
        0,
        0,
        loadings[i, 0] * 3,
        loadings[i, 1] * 3,
        color='r',
        alpha=0.8,
        head_width=0.1,
    )
    plt.text(
        loadings[i, 0] * 3.3,
        loadings[i, 1] * 3.3,
        feature,
        color='red',
        ha='center',
        va='center',
    )

  var1, var2 = pca.explained_variance_ratio_[:2]
  plt.xlabel(f'PC1 ({var1:.1%})')
  plt.ylabel(f'PC2 ({var2:.1%})')
  plt.title('Biplot PCA - Respuesta Enzimática FDA')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig('biplot_pca.png', dpi=300)
  plt.show()
