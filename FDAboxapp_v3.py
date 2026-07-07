import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import gc
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

# =========================================================================
# CONFIGURACIÓN E INICIALIZACIÓN DEL ESTADO
# =========================================================================
st.set_page_config(page_title="Análisis FDA - Cinéticas Completas", layout="wide")
st.title("🔬 Herramienta de Análisis FDA (Sensor de Suelo)")

if "procesado" not in st.session_state:
    st.session_state.procesado = False
    st.session_state.df_resultados = None

# =========================================================================
# BARRA LATERAL (CONTROLES DE PARÁMETROS)
# =========================================================================
st.sidebar.header("⚙️ Configuración del Análisis")
opcion_rotar = st.sidebar.selectbox("Rotación de Cámara:", ["Sin Rotación", "180 Grados", "90 Grados Horario", "90 Grados Antihorario"])
dict_rotacion = {"Sin Rotación": None, "180 Grados": cv2.ROTATE_180, "90 Grados Horario": cv2.ROTATE_90_CLOCKWISE, "90 Grados Antihorario": cv2.ROTATE_90_COUNTERCLOCKWISE}
rotacion_seleccionada = dict_rotacion[opcion_rotar]

if "ancho_px" not in st.session_state: st.session_state.ancho_px = 1280
if "alto_px" not in st.session_state: st.session_state.alto_px = 960

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Filtros S-G: Detección Espacial")
prop_sg_y = st.sidebar.slider("Ventana S-G en Y (%):", 0.1, 10.0, 1.5, step=0.1)
w_sg_y = max(5, int((prop_sg_y / 100.0) * st.session_state.alto_px))
if w_sg_y % 2 == 0: w_sg_y += 1

prop_sg_x = st.sidebar.slider("Ventana S-G en X (%):", 0.5, 10.0, 0.5, step=0.1)
w_sg_x = max(5, int((prop_sg_x / 100.0) * st.session_state.ancho_px))
if w_sg_x % 2 == 0: w_sg_x += 1

poly_sg = st.sidebar.slider("Orden Polinomio Detección:", 2, 5, 4)
factor_reduccion = st.sidebar.slider("Factor reducción ROI:", 0.0, 0.45, 0.20, step=0.05)

# =========================================================================
# 1. CARGA DE IMÁGENES SECUENCIALES
# =========================================================================
st.subheader("🗂️ 1. Carga de Imágenes Secuenciales")
archivos_subidos = st.file_uploader("Arrastrá tus fotos aquí", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if archivos_subidos:
    archivos_ordenados = sorted(archivos_subidos, key=lambda x: x.name)
    num_img = len(archivos_ordenados)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Rastreo Dinámico de ROIs")
    freq_roi = st.sidebar.slider("Frec. re-cálculo ROI (cada N fotos):", 1, num_img, num_img)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Filtros S-G: Cinéticas Temporales")
    w_sg_cin = st.sidebar.slider(f"Ventana S-G Temporal (5 a {num_img}):", min_value=5, max_value=max(5, num_img), value=min(31, max(5, num_img)), step=2)
    poly_sg_cin = st.sidebar.slider("Orden Polinomio Derivada:", 1, 5, 2)
    
    # =========================================================================
    # 2. DIAGNÓSTICO VISUAL Y CALIBRACIÓN DINÁMICA
    # =========================================================================
    st.markdown("---")
    st.subheader("📐 2. Diagnóstico Visual y Rastreo Dinámico")
    
    ref_indices = list(range(0, num_img, freq_roi))
    st.session_state.rois_por_ref = {}
    n_rois_base = 0
    w_roi_fijo, h_roi_fijo = 30, 20
    
    st.write(f"Se re-calcularán las coordenadas ROI en {len(ref_indices)} imágenes de referencia.")
    galeria_cols = st.columns(4)
    
    try:
        for idx_panel, i in enumerate(ref_indices):
            archivo = archivos_ordenados[i]
            img_bytes = archivo.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            archivo.seek(0)
            if rotacion_seleccionada is not None: img_bgr = cv2.rotate(img_bgr, rotacion_seleccionada)
            
            st.session_state.alto_px, st.session_state.ancho_px = img_bgr.shape[:2]
            canal_azul = img_bgr[:, :, 0]
            
            # Derivadas Y
            perfil_y = np.mean(canal_azul, axis=1)
            derivada_y = savgol_filter(perfil_y, window_length=w_sg_y, polyorder=poly_sg, deriv=1)
            y_min = int(np.argmax(derivada_y))
            y_max = int(np.argmin(derivada_y[y_min:]) + y_min)
            y_central = (y_min + y_max) // 2
            alto_banda = y_max - y_min

            # Derivadas X
            franja_azul = canal_azul[y_min:y_max, :]
            perfil_x = np.mean(franja_azul, axis=0)
            derivada_x = savgol_filter(perfil_x, window_length=w_sg_x, polyorder=poly_sg, deriv=1)
            umbral_x = np.max(np.abs(derivada_x)) * 0.15
            bordes_izq, _ = find_peaks(derivada_x, height=umbral_x, distance=max(15, st.session_state.ancho_px // 90))
            bordes_der, _ = find_peaks(-derivada_x, height=umbral_x, distance=max(15, st.session_state.ancho_px // 90))

            lista_centros_x = []
            for b_izq in bordes_izq:
                b_der_cands = bordes_der[bordes_der > b_izq]
                if len(b_der_cands) > 0:
                    b_der = b_der_cands[0]
                    if int(st.session_state.ancho_px * 0.008) < (b_der - b_izq) < int(st.session_state.ancho_px * 0.08):
                        lista_centros_x.append((b_izq, b_der, (b_izq + b_der) // 2))

            lista_centros_x.sort(key=lambda x: x[0])
            
            if i == 0:
                n_rois_base = len(lista_centros_x)
                if n_rois_base > 0:
                    ancho_min = min([b[1] - b[0] for b in lista_centros_x])
                    red_px = int(ancho_min * factor_reduccion * 2)
                    w_roi_fijo, h_roi_fijo = ancho_min - red_px, alto_banda - red_px
                
                st.session_state.rois_por_ref[i] = (lista_centros_x, y_central)
                
                st.markdown("**Gráficas de Diagnóstico (Señal cruda vs S-G) - Basadas en la primera foto:**")
                col_diag1, col_diag2 = st.columns(2)
                
                with col_diag1:
                    fig_y, ax1_y = plt.subplots(figsize=(5, 3))
                    ax1_y.plot(perfil_y, color='green', label='Intensidad')
                    ax1_y.set_xlabel("Posición Pixel Y"); ax1_y.set_ylabel("Intensidad", color='green')
                    ax1_y.tick_params(axis='y', labelcolor='green')
                    
                    ax2_y = ax1_y.twinx()
                    ax2_y.plot(derivada_y, color='red', alpha=0.7, label='1ª Derivada S-G')
                    ax2_y.set_ylabel("Gradiente (S-G)", color='red')
                    ax2_y.tick_params(axis='y', labelcolor='red')
                    
                    ax1_y.axvline(y_min, color='blue', linestyle='--', label=f'Borde ({y_min})')
                    ax1_y.axvline(y_max, color='blue', linestyle='--')
                    ax1_y.axvline(y_central, color='orange', linestyle='-', label=f'Centro ({y_central})')
                    
                    fig_y.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85), fontsize=7)
                    plt.title("Segmentación Eje Y")
                    st.pyplot(fig_y); plt.close(fig_y)
                    
                with col_diag2:
                    fig_x, ax1_x = plt.subplots(figsize=(5, 3))
                    ax1_x.plot(perfil_x, color='green', label='Intensidad')
                    ax1_x.set_xlabel("Posición Pixel X"); ax1_x.set_ylabel("Intensidad", color='green')
                    ax1_x.tick_params(axis='y', labelcolor='green')
                    
                    ax2_x = ax1_x.twinx()
                    ax2_x.plot(derivada_x, color='red', alpha=0.7, label='1ª Derivada S-G')
                    ax2_x.set_ylabel("Gradiente (S-G)", color='red')
                    ax2_x.tick_params(axis='y', labelcolor='red')
                    
                    if len(lista_centros_x) > 0:
                        ax1_x.axvline(lista_centros_x[0][0], color='blue', linestyle='--', label='Bordes')
                        ax1_x.axvline(lista_centros_x[0][1], color='blue', linestyle='--')
                        for c in lista_centros_x: ax1_x.axvline(c[2], color='orange', linestyle='-', alpha=0.5)
                        
                    fig_x.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85), fontsize=7)
                    plt.title("Detección de Flancos Eje X")
                    st.pyplot(fig_x); plt.close(fig_x)
            else:
                if len(lista_centros_x) != n_rois_base:
                    lista_centros_x, y_central = st.session_state.rois_por_ref[0]
                st.session_state.rois_por_ref[i] = (lista_centros_x, y_central)

            img_mascara = img_bgr.copy()
            for r_idx, (b_izq, b_der, cx) in enumerate(st.session_state.rois_por_ref[i][0]):
                y_c = st.session_state.rois_por_ref[i][1]
                cv2.rectangle(img_mascara, (b_izq, y_c - alto_banda//2), (b_der, y_c + alto_banda//2), (0, 255, 0), 2)
                x1_roi, x2_roi = cx - w_roi_fijo // 2, cx + w_roi_fijo // 2
                y1_roi, y2_roi = y_c - h_roi_fijo // 2, y_c + h_roi_fijo // 2
                cv2.rectangle(img_mascara, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 215, 255), 1)
                cv2.putText(img_mascara, f"R{r_idx+1}", (b_izq, y_c - alto_banda//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            img_resized = cv2.resize(img_mascara, (0,0), fx=0.3, fy=0.3)
            with galeria_cols[idx_panel % 4]:
                st.image(img_resized, channels="BGR", caption=f"Ref: {archivo.name}")
                
            del img_bgr, canal_azul, franja_azul, img_mascara, img_resized; gc.collect()
            
        st.session_state.w_roi_fijo = w_roi_fijo
        st.session_state.h_roi_fijo = h_roi_fijo
        st.session_state.n_rois_base = n_rois_base

    except Exception as e:
        st.error(f"Error en rastreo de ROIs: {e}"); st.stop()

    # =========================================================================
    # 3. MENÚ DE MUESTRAS
    # =========================================================================
    st.markdown("---")
    st.subheader("🏷️ 3. Identificación de Muestras y Metadatos Analíticos")
    
    nombres_muestras = []
    masas_muestras = []

    with st.expander("📝 Formulario de Carga: Nombres y Masas", expanded=True):
        c_h1, c_h2, c_h3 = st.columns([1, 2, 2])
        c_h1.markdown("**Código Base**")
        c_h2.markdown("**Nombre Asignado de la Muestra**")
        c_h3.markdown("**Masa Pesada (g)**")
        
        for i in range(st.session_state.n_rois_base):
            c_id, c_name, c_mass = st.columns([1, 2, 2])
            c_id.write(f"**ROI {i+1}**")
            
            key_name, key_mass = f"roi_name_val_{i}", f"roi_mass_val_{i}"
            if key_name not in st.session_state: st.session_state[key_name] = f"Muestra_{i+1}"
            if key_mass not in st.session_state: st.session_state[key_mass] = 1.0000
                
            val_name = c_name.text_input(f"Label {i+1}", value=st.session_state[key_name], label_visibility="collapsed", key=f"ui_str_{i}")
            val_mass = c_mass.number_input(f"Mass {i+1}", value=st.session_state[key_mass], min_value=0.0000, step=0.0001, format="%.4f", label_visibility="collapsed", key=f"ui_num_{i}")
            
            st.session_state[key_name] = val_name; st.session_state[key_mass] = val_mass
            nombres_muestras.append(val_name); masas_muestras.append(val_mass)

    # =========================================================================
    # 4. PROCESAMIENTO KINÉTICO POR LOTE
    # =========================================================================
    st.markdown("---")
    st.subheader("🚀 4. Ejecución del Perfil Cinético Completo")
    
    if st.button("▶️ Lanzar Procesamiento de Lote"):
        if st.session_state.n_rois_base > 0:
            barra_progreso = st.progress(0)
            
            tiempos_dt = []
            for f in archivos_ordenados:
                try: tiempos_dt.append(datetime.strptime(f.name.rsplit('.', 1)[0], "%Y-%m-%d_%H-%M-%S"))
                except ValueError: st.error(f"Falla de parseo: {f.name} no es AAAA-MM-DD_HH-mm-ss"); st.stop()
                    
            t0 = min(tiempos_dt)
            t_rel_min = np.array([(t - t0).total_seconds() / 60.0 for t in tiempos_dt])

            n_r = st.session_state.n_rois_base
            h_verde, h_azul, h_ratios = np.zeros((num_img, n_r)), np.zeros((num_img, n_r)), np.zeros((num_img, n_r))

            for idx, archivo in enumerate(archivos_ordenados):
                barra_progreso.progress(int((idx + 1) / num_img * 100))
                
                ref_idx = (idx // freq_roi) * freq_roi
                centros_tuplas, y_cent_actual = st.session_state.rois_por_ref[ref_idx]
                centros_x_act = [b[2] for b in centros_tuplas]
                
                img_bytes = archivo.read()
                frame_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if rotacion_seleccionada is not None: frame_bgr = cv2.rotate(frame_bgr, rotacion_seleccionada)
                
                c_a, c_v = frame_bgr[:, :, 0], frame_bgr[:, :, 1]
                w_r, h_r = st.session_state.w_roi_fijo, st.session_state.h_roi_fijo
                
                for r_idx, cx in enumerate(centros_x_act):
                    x1, x2 = cx - w_r // 2, cx + w_r // 2
                    y1, y2 = y_cent_actual - h_r // 2, y_cent_actual + h_r // 2
                    m_v, m_a = np.mean(c_v[y1:y2, x1:x2]), np.mean(c_a[y1:y2, x1:x2])
                    h_verde[idx, r_idx] = m_v; h_azul[idx, r_idx] = m_a
                    h_ratios[idx, r_idx] = m_v / (m_a if m_a > 0 else 1.0)
                
                # ACÁ ESTABA EL ERROR. AHORA SÍ EN DOS LÍNEAS DISTINTAS:
                del frame_bgr, c_a, c_v
                if idx % 10 == 0: 
                    gc.collect()

            w_cin = w_sg_cin if w_sg_cin <= num_img else (num_img if num_img % 2 != 0 else num_img - 1)
            poly_cin = poly_sg_cin if poly_sg_cin < w_cin else w_cin - 1
            margen = w_cin // 2
            
            h_deriv_verde = np.zeros_like(h_verde); h_deriv_azul = np.zeros_like(h_azul); h_deriv_ratios = np.zeros_like(h_ratios)
            
            for r in range(n_r):
                dv = savgol_filter(h_verde[:, r], window_length=w_cin, polyorder=poly_cin, deriv=1)
                da = savgol_filter(h_azul[:, r], window_length=w_cin, polyorder=poly_cin, deriv=1)
                dr = savgol_filter(h_ratios[:, r], window_length=w_cin, polyorder=poly_cin, deriv=1)
                
                dv[:margen], dv[-margen:] = np.nan, np.nan
                da[:margen], da[-margen:] = np.nan, np.nan
                dr[:margen], dr[-margen:] = np.nan, np.nan
                
                h_deriv_verde[:, r], h_deriv_azul[:, r], h_deriv_ratios[:, r] = dv, da, dr

            cols = ["Nombre_Archivo", "Tiempo_Rel_Minutos"]
            for r in range(n_r):
                cols.extend([f"{nombres_muestras[r]}_Verde", f"{nombres_muestras[r]}_dG/dt", 
                             f"{nombres_muestras[r]}_Azul", f"{nombres_muestras[r]}_dB/dt", 
                             f"{nombres_muestras[r]}_Ratio", f"{nombres_muestras[r]}_dRatio/dt", f"{nombres_muestras[r]}_Masa_g"])
                
            datos = []
            for i in range(num_img):
                fila = [archivos_ordenados[i].name, t_rel_min[i]]
                for r in range(n_r):
                    fila.extend([h_verde[i,r], h_deriv_verde[i,r], h_azul[i,r], h_deriv_azul[i,r], h_ratios[i,r], h_deriv_ratios[i,r], masas_muestras[r]])
                datos.append(fila)

            st.session_state.df = pd.DataFrame(datos, columns=cols)
            st.session_state.t_rel_min = t_rel_min
            st.session_state.h_verde, st.session_state.h_azul, st.session_state.h_ratios = h_verde, h_azul, h_ratios
            st.session_state.h_dverde, st.session_state.h_dazul, st.session_state.h_dratio = h_deriv_verde, h_deriv_azul, h_deriv_ratios
            st.session_state.nombres_finales, st.session_state.masas_finales = nombres_muestras, masas_muestras
            
            reporte_str = f"""========================================
REPORTE DE CONFIGURACIÓN - ANÁLISIS FDA
========================================
Fecha de Análisis: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total de Imágenes Procesadas: {num_img}
Resolución de Imágenes (Alto x Ancho): {st.session_state.alto_px} x {st.session_state.ancho_px} px

[PARÁMETROS ESPACIALES Y DE ROI]
Frecuencia de re-cálculo de ROI: Cada {freq_roi} imágenes
Ventana S-G (Y): {prop_sg_y}% ({w_sg_y} px)
Ventana S-G (X): {prop_sg_x}% ({w_sg_x} px)
Polinomio S-G (Detección): {poly_sg}
Tamaño final de ROI (Ancho x Alto): {w_r} x {h_r} px

[PARÁMETROS CINÉTICOS]
Ventana S-G Temporal: {w_sg_cin}
Polinomio S-G Temporal: {poly_sg_cin}

[IDENTIFICACIÓN DE MUESTRAS (N={n_r})]
"""
            for i in range(n_r): reporte_str += f"- ROI {i+1} | Nombre: {nombres_muestras[i]} | Masa: {masas_muestras[i]:.4f} g\n"
            st.session_state.reporte_txt = reporte_str
            st.session_state.procesado = True

    # =========================================================================
    # 5. RENDERIZADO DE RESULTADOS FINALES
    # =========================================================================
    if st.session_state.procesado:
        t = st.session_state.t_rel_min
        lbls = st.session_state.nombres_finales
        plt.close('all')

        st.markdown("---")
        st.subheader("📊 Dinámicas Espectrales (Señales Crudas)")
        c1, c2, c3 = st.columns(3)
        with c1:
            fig, ax = plt.subplots(figsize=(5,4)); 
            for r in range(st.session_state.n_rois_base): ax.plot(t, st.session_state.h_verde[:, r], label=lbls[r])
            ax.set_title("Verde vs Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7); st.pyplot(fig); plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5,4)); 
            for r in range(st.session_state.n_rois_base): ax.plot(t, st.session_state.h_azul[:, r], label=lbls[r])
            ax.set_title("Azul vs Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7); st.pyplot(fig); plt.close(fig)
        with c3:
            fig, ax = plt.subplots(figsize=(5,4)); 
            for r in range(st.session_state.n_rois_base): ax.plot(t, st.session_state.h_ratios[:, r], label=lbls[r])
            ax.set_title("Ratio vs Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7); st.pyplot(fig); plt.close(fig)

        st.markdown("---")
        st.subheader("📈 Tasas de Reacción (1ª Derivada S-G)")
        d1, d2, d3 = st.columns(3)
        with d1:
            fig, ax = plt.subplots(figsize=(5,4)); 
            for r in range(st.session_state.n_rois_base): ax.plot(t, st.session_state.h_dverde[:, r], label=lbls[r])
            ax.set_title("dVerde/dt vs Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7); st.pyplot(fig); plt.close(fig)
        with d2:
            fig, ax = plt.subplots(figsize=(5,4)); 
            for r in range(st.session_state.n_rois_base): ax.plot(t, st.session_state.h_dazul[:, r], label=lbls[r])
            ax.set_title("dAzul/dt vs Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7); st.pyplot(fig); plt.close(fig)
        with d3:
            fig, ax = plt.subplots(figsize=(5,4)); 
            for r in range(st.session_state.n_rois_base): ax.plot(t, st.session_state.h_dratio[:, r], label=lbls[r])
            ax.set_title("dRatio/dt vs Minutos"); ax.grid(True, alpha=0.3); ax.legend(fontsize=7); st.pyplot(fig); plt.close(fig)

        st.markdown("---")
        st.subheader("💾 Descarga de Resultados")
        down_col1, down_col2 = st.columns(2)
        with down_col1:
            st.download_button("📥 Descargar Tabla (CSV)", st.session_state.df.to_csv(index=False).encode('utf-8'), f"cinetica_fda_{int(time.time())}.csv", "text/csv")
        with down_col2:
            st.download_button("📝 Descargar Reporte de Configuración (TXT)", st.session_state.reporte_txt.encode('utf-8'), f"reporte_config_{int(time.time())}.txt", "text/plain")
            
        st.dataframe(st.session_state.df.head(10).style.format(precision=4, na_rep='NaN'), use_container_width=True)
