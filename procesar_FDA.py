import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.integrate import simpson
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. PARSER DE ARCHIVOS Y METADATOS
# ==========================================

DIR_INPUT = "datos_raw"
DIR_OUTPUT = "resultados"
os.makedirs(DIR_OUTPUT, exist_ok=True)

def parse_filename(filename):
    """Extrae [ENSAYO]_[DIA]_[CULTIVO]_[CORRIDA].csv"""
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("_")
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], parts[3]
    return "FDA", parts[0], "general", "1"

def parse_col_header(col_name):
    """Extrae Condición, Posición, Maceta_ID y Variable de encabezados tipo 'P1 (46) _G'"""
    m = re.match(r"^([A-Za-z]\d+)\s*\((.*?)\)\s*_?(.*)$", col_name.strip())
    if m:
        pos_str = m.group(1)
        return {
            'Condicion': pos_str[0],
            'Posicion': int(pos_str[1:]),
            'Maceta_ID': m.group(2).strip(),
            'Variable': m.group(3).strip().replace('-', '_').replace('/', '_')
        }
    return None

def cargar_y_limpiar_datos():
    archivos = glob.glob(os.path.join(DIR_INPUT, "*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos .csv en la carpeta '{DIR_INPUT}'")
        
    registros = []
    
    for filepath in archivos:
        fname = os.path.basename(filepath)
        ensayo, dia, cultivo, corrida = parse_filename(fname)
        df_raw = pd.read_csv(filepath)
        
        time_col = [c for c in df_raw.columns if 'tiempo' in c.lower() or 'time' in c.lower()][0]
        parsed_cols = {c: parse_col_header(c) for c in df_raw.columns if c != time_col and parse_col_header(c)}
        
        muestras = set((v['Condicion'], v['Posicion'], v['Maceta_ID']) for v in parsed_cols.values())
        
        for cond, pos, maceta in muestras:
            cols_muestra = [k for k, v in parsed_cols.items() 
                            if v['Condicion']==cond and v['Posicion']==pos and v['Maceta_ID']==maceta]
            
            masa_col = [k for k in cols_muestra if parsed_cols[k]['Variable'].lower() == 'masa']
            masa_val = df_raw[masa_col[0]].dropna().iloc[0] if masa_col and len(df_raw[masa_col[0]].dropna()) > 0 else 1.0
            
            sub = pd.DataFrame({
                'Archivo': fname, 'Ensayo': ensayo, 'Dia_Cultivo': dia,
                'Cultivo': cultivo, 'Corrida': corrida, 'Condicion': cond,
                'Posicion': pos, 'Maceta_ID': maceta, 'Masa': masa_val,
                'Tiempo': df_raw[time_col]
            })
            
            for k in cols_muestra:
                var_name = parsed_cols[k]['Variable']
                if var_name.lower() != 'masa':
                    sub[var_name] = df_raw[k]
                    
            registros.append(sub)
            
    return pd.concat(registros, ignore_index=True)

# ==========================================
# 2. CARACTERÍSTICAS Y DERIVADAS (S-G)
# ==========================================

def procesar_cineticas(df_tidy, t_corte=15.0):
    caracteristicas = []
    ts_procesadas = []

    grupos = df_tidy.groupby(['Archivo', 'Ensayo', 'Dia_Cultivo', 'Cultivo', 'Corrida', 'Condicion', 'Posicion', 'Maceta_ID'])

    for keys, group in grupos:
        group = group.sort_values('Tiempo').reset_index(drop=True)
        t = group['Tiempo'].values
        masa = keys[8] if len(keys) > 8 else group['Masa'].iloc[0]
        dt = np.median(np.diff(t))

        # Señales Netas
        if 'G_G0' not in group.columns and 'G' in group.columns:
            group['G_G0'] = group['G'] - group['G'].iloc[0]
        if 'A_A0' not in group.columns and 'A' in group.columns:
            group['A_A0'] = group['A'] - group['A'].iloc[0]
        if 'R_R0' not in group.columns and 'R' in group.columns:
            group['R_R0'] = group['R'] / group['R'].iloc[0] if group['R'].iloc[0] != 0 else group['R']

        # Filtro Savitzky-Golay
        win = 11 if len(group) >= 11 else (len(group)-1 if (len(group)-1)%2!=0 else len(group)-2)
        group['dG_dt'] = savgol_filter(group['G_G0'], window_length=win, polyorder=2, deriv=1, delta=dt)
        group['dA_dt'] = savgol_filter(group['A_A0'], window_length=win, polyorder=2, deriv=1, delta=dt)
        group['dR_dt'] = savgol_filter(group['R_R0'], window_length=win, polyorder=2, deriv=1, delta=dt)
        group['d2G_dt2'] = savgol_filter(group['G_G0'], window_length=win, polyorder=2, deriv=2, delta=dt)

        # Velocidades Normalizadas
        group['Vel_G_norm'] = group['dG_dt'] / masa
        group['Vel_A_norm'] = group['dA_dt'] / masa
        group['Vel_R_norm'] = group['dR_dt'] / masa

        ts_procesadas.append(group)

        # Truncamiento t > 15
        mask = t > t_corte
        t_sub = t[mask]
        dG_sub = group['dG_dt'].values[mask]
        dA_sub = group['dA_dt'].values[mask]
        d2G_sub = group['d2G_dt2'].values[mask]
        G_sub = group['G_G0'].values[mask]

        if len(t_sub) > 2:
            idx_vmax = np.argmax(dG_sub)
            vmax = dG_sub[idx_vmax]
            t_vmax = t_sub[idx_vmax]
            auc_g = simpson(y=G_sub, x=t_sub)
            decay_azul = np.max(dA_sub) - dA_sub[-1]
            concavidad = np.mean(d2G_sub[idx_vmax:]) if idx_vmax < len(d2G_sub) else d2G_sub[idx_vmax]

            caracteristicas.append({
                'Archivo': keys[0], 'Ensayo': keys[1], 'Dia_Cultivo': keys[2],
                'Cultivo': keys[3], 'Corrida': keys[4], 'Condicion': keys[5],
                'Posicion': keys[6], 'Maceta_ID': keys[7], 'Masa': masa,
                'Vmax_G_norm': vmax / masa, 't_Vmax_G': t_vmax,
                'AUC_G_norm': auc_g / masa, 'Decay_Azul_norm': decay_azul / masa,
                'Concavidad_post_Vmax': concavidad
            })

    return pd.concat(ts_procesadas, ignore_index=True), pd.DataFrame(caracteristicas)

# ==========================================
# 3. REPORTE INTERACTIVO (PLOTLY)
# ==========================================

def generar_reporte_plotly(df_ts):
    fig = make_subplots(rows=2, cols=3, subplot_titles=(
        'Intensidad Verde (G-G0)', 'Intensidad Azul (A-A0)', 'Ratio Rojo (R/R0)',
        'Velocidad Verde Norm', 'Velocidad Azul Norm', 'Velocidad Ratio Rojo Norm'
    ))
    
    métricas = ['G_G0', 'A_A0', 'R_R0', 'Vel_G_norm', 'Vel_A_norm', 'Vel_R_norm']
    condiciones = df_ts['Condicion'].unique()

    for idx, metrica in enumerate(métricas):
        r, c = (idx // 3) + 1, (idx % 3) + 1
        stats = df_ts.groupby(['Condicion', 'Tiempo'])[metrica].agg(['mean', 'std']).reset_index()
        
        for cond in condiciones:
            sub = stats[stats['Condicion'] == cond]
            fig.add_trace(go.Scatter(
                x=sub['Tiempo'], y=sub['mean'], mode='lines', name=f'Cond {cond}',
                legendgroup=f'Cond {cond}', showlegend=(idx == 0)
            ), row=r, col=c)

    fig.update_layout(height=750, title_text="Cinéticas Enzimáticas FDA - Reporte Interactivo")
    fig.write_html(os.path.join(DIR_OUTPUT, "reporte_interactivo_fda.html"))

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    df_tidy = cargar_y_limpiar_datos()
    df_ts, df_features = procesar_cineticas(df_tidy)

    # 1. PCA (Se ejecuta si hay suficientes muestras)
    cols_feat = ['Vmax_G_norm', 'AUC_G_norm', 'Decay_Azul_norm', 'Concavidad_post_Vmax', 't_Vmax_G']
    if len(df_features) >= 2:
        X = StandardScaler().fit_transform(df_features[cols_feat].fillna(0))
        n_comp = min(2, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_comp).fit_transform(X)
        df_features['PC1'] = pca[:, 0]
        if n_comp > 1:
            df_features['PC2'] = pca[:, 1]

    # 2. Modelo Estadístico Adaptativo (Evita la matriz singular)
    factores = []
    if df_features['Condicion'].nunique() > 1:
        factores.append("C(Condicion)")
    if df_features['Dia_Cultivo'].nunique() > 1:
        factores.append("C(Dia_Cultivo)")

    if factores:
        formula = f"Vmax_G_norm ~ {' + '.join(factores)}"
        try:
            if df_features['Posicion'].nunique() > 1:
                lmm = smf.mixedlm(formula, df_features, groups=df_features["Posicion"]).fit()
            else:
                lmm = smf.ols(formula, df_features).fit()
            print("--- Resumen del Modelo Estadístico ---")
            print(lmm.summary())
        except Exception as e:
            print(f"No se pudo evaluar el modelo lineal: {e}")
    else:
        print("Aviso: Se requiere más de un Día o Condición para ejecutar el análisis ANOVA/LMM.")

    # 3. Exportación
    df_ts.to_csv(os.path.join(DIR_OUTPUT, "fda_tidy_dataset.csv"), index=False)
    df_features.to_excel(os.path.join(DIR_OUTPUT, "fda_features_summary.xlsx"), index=False)
    generar_reporte_plotly(df_ts)

    print("\nProcesamiento finalizado con éxito. Resultados guardados en 'resultados/'.")
