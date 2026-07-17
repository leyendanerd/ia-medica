"""
DentalVision AI — MVP
Sistema de IA para diagnósticos dentales — ejecución local (Edge AI)

Alineado con el proyecto de innovación (UNIR — Maestría en IA):
  - Numeración FDI automática de piezas (YOLOv11)
  - Detección de caries, pérdida ósea y quistes (Mask R-CNN)
  - Generación automática de odontograma
  - Procesamiento 100% local — sin conexión a internet
  - Diseñado para desplegar en NVIDIA Jetson Orin Nano (TensorRT/TFLite)
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import io
import time
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.image_processor import ImageProcessor
from utils.dicom_handler import DicomHandler
from utils.mock_inference import MockDentalInference, PATHOLOGY_CLASSES
from utils.odontogram_renderer import render_odontogram_svg
from utils.auth import require_login, render_logout_control

st.set_page_config(page_title="DentalVision AI", page_icon="🦷", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg-primary: #0a0e1a; --bg-card: #111827; --bg-elevated: #1a2235;
    --accent-cyan: #00d4e8; --accent-blue: #4f8ef7; --accent-red: #ff4757;
    --accent-green: #2ed573; --accent-yellow: #ffa502; --accent-purple: #a259ff;
    --text-primary: #e8f0fe; --text-secondary: #8899bb; --border: #1e2d4a;
}
.stApp {
    background: var(--bg-primary);
    background-image: radial-gradient(ellipse at 20% 20%, rgba(79,142,247,0.08) 0%, transparent 50%),
                       radial-gradient(ellipse at 80% 80%, rgba(0,212,232,0.06) 0%, transparent 50%);
    font-family: 'Space Grotesk', sans-serif; color: var(--text-primary);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; max-width: 1440px; }

.dental-header {
    display: flex; align-items: center; gap: 1rem; padding: 1.25rem 2rem;
    background: linear-gradient(135deg, #0d1526 0%, #111f3a 100%);
    border: 1px solid var(--border); border-radius: 16px; margin-bottom: 1.25rem;
    position: relative; overflow: hidden;
}
.dental-header::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-blue), transparent);
}
.header-icon { font-size: 2.3rem; }
.header-title {
    font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 800;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}
.header-subtitle { font-size: 0.82rem; color: var(--text-secondary); margin: 0; }
.header-badges { margin-left: auto; display: flex; gap: 0.5rem; }
.badge { font-size: 0.7rem; font-weight: 600; padding: 0.3rem 0.7rem; border-radius: 20px; letter-spacing: 0.03em; }
.badge-cyan { background: rgba(0,212,232,0.1); border: 1px solid rgba(0,212,232,0.3); color: var(--accent-cyan); }
.badge-green { background: rgba(46,213,115,0.1); border: 1px solid rgba(46,213,115,0.3); color: var(--accent-green); }

.card-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-secondary); margin-bottom: 0.75rem; }

.upload-zone {
    border: 2px dashed var(--border); border-radius: 16px; padding: 2.5rem 2rem; text-align: center;
    background: rgba(79,142,247,0.03);
}

.metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; margin-bottom: 1rem; }
.metric-card { background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; text-align: center; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 700; line-height: 1.1; }
.metric-label { font-size: 0.68rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.25rem; }
.m-cyan { color: var(--accent-cyan); } .m-red { color: var(--accent-red); }
.m-green { color: var(--accent-green); } .m-amber { color: var(--accent-yellow); } .m-purple { color: var(--accent-purple); }

.pathology-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-bottom: 1rem; }
.path-card { background: var(--bg-elevated); border-radius: 10px; padding: 0.6rem; text-align: center; }
.path-num { font-size: 1.2rem; font-weight: 700; font-family: 'Syne', sans-serif; }
.path-lbl { font-size: 0.62rem; color: var(--text-secondary); margin-top: 2px; }

.severity-bar { height: 6px; border-radius: 3px; background: var(--bg-elevated); margin-top: 0.5rem; overflow: hidden; }
.severity-fill { height: 100%; border-radius: 3px; transition: width 0.8s ease; }

.detection-item { display: flex; align-items: center; gap: 0.75rem; padding: 0.55rem 0.75rem; background: var(--bg-elevated); border-radius: 8px; margin-bottom: 0.4rem; font-size: 0.83rem; }
.detection-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.detection-label { flex: 1; color: var(--text-primary); }
.detection-confidence { font-size: 0.73rem; color: var(--text-secondary); }

.status-live { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; color: var(--accent-green); }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent-green); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1);} 50% { opacity:0.5; transform:scale(1.3);} }

.analysis-progress { background: var(--bg-elevated); border-radius: 4px; height: 4px; overflow: hidden; margin: 0.75rem 0; }
.analysis-fill { height: 100%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); animation: loading 1.5s ease-in-out infinite; }
@keyframes loading { 0% { transform: translateX(-100%);} 100% { transform: translateX(100%);} }

section[data-testid="stSidebar"] { background: var(--bg-card) !important; border-right: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] .stSelectbox label, section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label { color: var(--text-secondary) !important; font-size: 0.8rem !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: var(--bg-primary) !important; font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
    border: none !important; border-radius: 10px !important; padding: 0.6rem 1.5rem !important; font-size: 0.9rem !important;
    transition: opacity 0.2s !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

img { border-radius: 10px; }

.stTabs [data-baseweb="tab-list"] { background: var(--bg-card); border-radius: 10px; padding: 0.25rem; gap: 0.25rem; }
.stTabs [data-baseweb="tab"] { background: transparent; color: var(--text-secondary) !important; border-radius: 8px; font-size: 0.85rem; font-weight: 500; }
.stTabs [aria-selected="true"] { background: var(--bg-elevated) !important; color: var(--text-primary) !important; }

hr { border-color: var(--border) !important; opacity: 0.5; }
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: var(--bg-primary); } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.edge-banner {
    display: flex; align-items: center; gap: 0.6rem; padding: 0.6rem 1rem;
    background: rgba(46,213,115,0.06); border: 1px solid rgba(46,213,115,0.2); border-radius: 10px;
    font-size: 0.78rem; color: var(--accent-green); margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Autenticación ─────────────────────────────────────────────────────────────
current_user = require_login()

# ── Servicios cacheados ────────────────────────────────────────────────────────
@st.cache_resource
def load_inference_engine():
    return MockDentalInference()

@st.cache_resource
def load_image_processor():
    return ImageProcessor(target_size=(512, 512))

inference = load_inference_engine()
processor = load_image_processor()
dicom_handler = DicomHandler()

# ── Estado de sesión ──────────────────────────────────────────────────────────
for key in ("results", "processed_image", "original_image"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dental-header">
    <span class="header-icon">🦷</span>
    <div>
        <p class="header-title">DentalVision AI</p>
        <p class="header-subtitle">Detección de patologías dentales · Edge AI · Sin conexión a internet</p>
    </div>
    <div class="header-badges">
        <span class="badge badge-green">🔒 Procesamiento local</span>
        <span class="badge badge-cyan">MVP v0.2</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="edge-banner">
    ⚡ Arquitectura Edge AI — diseñada para NVIDIA Jetson Orin Nano · YOLOv11 (detección) + Mask R-CNN (segmentación) · Meta de latencia: &lt;5s por imagen
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    render_logout_control()
    st.markdown("---")

    st.markdown('<p class="card-title">⚙️ Configuración del modelo</p>', unsafe_allow_html=True)
    confidence_threshold = st.slider("Umbral de confianza", 0.30, 0.95, 0.50, 0.05)
    iou_threshold = st.slider("Umbral IOU (NMS)", 0.1, 0.9, 0.45, 0.05)

    show_heatmap = st.checkbox("Mostrar mapa de calor", value=True)
    show_boxes = st.checkbox("Mostrar bounding boxes", value=True)
    show_labels = st.checkbox("Mostrar etiquetas", value=True)
    heatmap_alpha = st.slider("Transparencia del heatmap", 0.2, 0.8, 0.45, 0.05)

    st.markdown("---")
    st.markdown('<p class="card-title">📊 Estado del sistema</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem; color:#8899bb; line-height:2;">
        <div class="status-live"><span class="status-dot"></span> Modelo cargado</div>
        <div style="margin-top:0.4rem;">🧠 YOLOv11 + Mask R-CNN (mock)</div>
        <div>📐 Entrada: 512×512 px</div>
        <div>⚡ Target: NVIDIA Jetson Orin Nano</div>
        <div>📦 Runtime: TensorFlow Lite / TensorRT</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="card-title">ℹ️ Formatos soportados</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#8899bb; line-height:1.9;">
        ✅ JPEG / JPG<br>✅ PNG<br>✅ DICOM (.dcm)<br>🔄 BMP, TIFF (conversión auto)
    </div>
    """, unsafe_allow_html=True)

# ── Layout principal ──────────────────────────────────────────────────────────
col_upload, col_results = st.columns([1.1, 1], gap="large")

with col_upload:
    st.markdown('<p class="card-title">📂 Cargar radiografía</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Arrastra o selecciona un archivo", type=["jpg", "jpeg", "png", "dcm"],
        help="Formatos: JPEG, PNG o DICOM (.dcm)", label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:2.3rem; margin-bottom:0.6rem;">🩻</div>
            <p style="font-size:1rem; font-weight:600; color:#e8f0fe; margin:0;">Arrastra tu radiografía panorámica aquí</p>
            <p style="font-size:0.8rem; color:#8899bb; margin:0.4rem 0 0;">JPG · PNG · DICOM — hasta 200 MB</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        file_bytes = uploaded_file.read()
        file_ext = Path(uploaded_file.name).suffix.lower()

        with st.spinner("Procesando imagen..."):
            if file_ext == ".dcm":
                pil_image = dicom_handler.load(file_bytes)
                meta = dicom_handler.extract_metadata(file_bytes)
                st.info("🏥 Archivo DICOM cargado correctamente", icon="ℹ️")
            else:
                pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                meta = {}

        # OPTIMIZACION DE MEMORIA: reducir la imagen apenas se carga.
        # El modelo trabaja a 512x512 de todas formas, asi que la resolucion
        # extra de una panoramica (2000x1000+) no aporta precision, solo consume
        # RAM: se mantendrian ~3 copias grandes en memoria (original, CLAHE,
        # anotada). Critico en VMs con poca RAM.
        MAX_DIM = 1024
        if max(pil_image.size) > MAX_DIM:
            pil_image.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)

        st.session_state.original_image = pil_image

        tab_orig, tab_proc = st.tabs(["Original", "Pre-procesada (CLAHE)"])
        with tab_orig:
            st.image(pil_image, use_column_width=True, caption=f"📁 {uploaded_file.name}")
        with tab_proc:
            processed = processor.preprocess(pil_image)
            st.image(processed["display"], use_column_width=True, caption="512×512 · normalizado · CLAHE (OpenCV, espacio LAB)")

        st.markdown(f"""
        <div class="card" style="margin-top:0.75rem; background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:0.9rem;">
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.5rem; font-size:0.76rem; text-align:center;">
                <div><div style="color:#8899bb;">Resolución</div><div style="font-weight:600;">{pil_image.width}×{pil_image.height}</div></div>
                <div><div style="color:#8899bb;">Formato</div><div style="font-weight:600;">{file_ext.upper().replace('.','')}</div></div>
                <div><div style="color:#8899bb;">Tamaño</div><div style="font-weight:600;">{len(file_bytes)/1024:.0f} KB</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍 Analizar radiografía", type="primary"):
            progress_placeholder = st.empty()
            progress_placeholder.markdown("""
            <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:1rem;">
                <div style="font-size:0.85rem; color:#8899bb; margin-bottom:0.5rem;">⚙️ Ejecutando inferencia (YOLOv11 + Mask R-CNN)...</div>
                <div class="analysis-progress"><div class="analysis-fill"></div></div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.1)

            results = inference.predict(
                pil_image, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold,
                show_heatmap=show_heatmap, show_boxes=show_boxes, show_labels=show_labels, heatmap_alpha=heatmap_alpha,
            )
            st.session_state.results = results
            progress_placeholder.empty()
            st.success(f"✅ Análisis completado en {results['inference_ms']:.0f} ms", icon="✅")

        # ── Envío al sistema clínico ─────────────────────────────────────────
        # Se habilita SOLO cuando ya existe un análisis (st.session_state.results).
        # Mientras no haya resultados, el botón aparece deshabilitado (disabled=True).
        analysis_done = st.session_state.results is not None

        if st.button(
            "📤 Enviar información a sistema clínico",
            disabled=not analysis_done,
            key="send_to_clinic",
            help="Disponible tras ejecutar el análisis" if not analysis_done else "Envía el reporte al sistema de gestión de la clínica",
        ):
            with st.spinner("Enviando al sistema clínico..."):
                time.sleep(1.0)  # Simula la latencia de una llamada al HIS/sistema clínico
            r = st.session_state.results
            st.success("✅ Información enviada correctamente al sistema clínico", icon="✅")
            st.markdown(f"""
            <div style="background:rgba(46,213,115,0.06); border:1px solid rgba(46,213,115,0.25);
                        border-radius:10px; padding:0.75rem 1rem; margin-top:0.5rem; font-size:0.8rem; color:#8899bb;">
                <div style="color:#2ed573; font-weight:600; margin-bottom:0.35rem;">📋 Registro clínico generado</div>
                <div>ID de envío: <span style="color:#e8f0fe;">REC-{time.strftime('%Y%m%d-%H%M%S')}</span></div>
                <div>Piezas registradas: <span style="color:#e8f0fe;">{r['total_teeth']}</span> ·
                     Hallazgos: <span style="color:#e8f0fe;">{r['affected_teeth']}</span></div>
                <div>Profesional: <span style="color:#e8f0fe;">{current_user['display_name']}</span></div>
                <div style="margin-top:0.35rem; font-size:0.72rem; opacity:0.7;">
                    ⚠️ Simulación (MVP) — no hay integración real con un HIS todavía.
                </div>
            </div>
            """, unsafe_allow_html=True)

with col_results:
    st.markdown('<p class="card-title">📊 Resultados del análisis</p>', unsafe_allow_html=True)

    if st.session_state.results is None:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1.5rem; background:var(--bg-card); border:1px solid var(--border); border-radius:12px;">
            <div style="font-size:2rem; margin-bottom:0.75rem;">📋</div>
            <p style="color:#8899bb; font-size:0.9rem; margin:0;">Los resultados aparecerán aquí<br>una vez que analices una radiografía</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        r = st.session_state.results

        tab_img, tab_odonto = st.tabs(["Radiografía anotada", "Odontograma automático"])

        with tab_img:
            if r.get("annotated_image"):
                st.image(r["annotated_image"], use_column_width=True, caption="Bounding boxes + heatmap de hallazgos")

        with tab_odonto:
            # El SVG se renderiza dentro de un iframe aislado (components.html)
            # en lugar de st.markdown(unsafe_allow_html=True): Streamlit sanitiza
            # el HTML inyectado por markdown y eso puede romper su renderizador React.
            svg = render_odontogram_svg(r["odontogram"])
            components.html(
                f"""
                <div style="background:#0a0e1a; padding:12px; border-radius:12px;
                            font-family: 'Space Grotesk', system-ui, sans-serif;">
                    {svg}
                </div>
                """,
                height=260,
                scrolling=False,
            )
            st.caption("Odontograma generado automáticamente en notación FDI (universal) — objetivo central del proyecto.")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        total = r["total_teeth"]
        affected = r["affected_teeth"]
        healthy = total - affected
        severity_pct = r["global_severity_pct"]

        if severity_pct < 30:
            sev_color, sev_label = "#2ed573", "BAJO"
        elif severity_pct < 60:
            sev_color, sev_label = "#ffa502", "MODERADO"
        else:
            sev_color, sev_label = "#ff4757", "ALTO"

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-value m-cyan">{total}</div><div class="metric-label">Piezas detectadas</div></div>
            <div class="metric-card"><div class="metric-value m-red">{affected}</div><div class="metric-label">Con hallazgos</div></div>
            <div class="metric-card"><div class="metric-value m-green">{healthy}</div><div class="metric-label">Sanas</div></div>
            <div class="metric-card"><div class="metric-value" style="color:{sev_color};">{severity_pct:.0f}%</div><div class="metric-label">Severidad global</div></div>
        </div>

        <div class="pathology-grid">
            <div class="path-card"><div class="path-num m-red">{r['pathology_counts']['caries']}</div><div class="path-lbl">Caries</div></div>
            <div class="path-card"><div class="path-num m-amber">{r['pathology_counts']['perdida_osea']}</div><div class="path-lbl">Pérdida ósea</div></div>
            <div class="path-card"><div class="path-num m-purple">{r['pathology_counts']['quiste']}</div><div class="path-lbl">Quistes</div></div>
        </div>

        <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:0.75rem 1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem;">
                <span style="color:#8899bb;">Nivel de riesgo</span>
                <span style="color:{sev_color}; font-weight:700; font-size:0.85rem;">{sev_label}</span>
            </div>
            <div class="severity-bar"><div class="severity-fill" style="width:{severity_pct}%; background:{sev_color};"></div></div>
        </div>
        """, unsafe_allow_html=True)

        pathology_detections = [d for d in r["detections"] if d["class"] != "sano"]
        if pathology_detections:
            st.markdown('<p class="card-title" style="margin-top:1rem;">🎯 Hallazgos clínicos</p>', unsafe_allow_html=True)
            for det in pathology_detections:
                color = PATHOLOGY_CLASSES[det["class"]]["color"]
                hex_color = "#%02x%02x%02x" % color[:3]
                st.markdown(f"""
                <div class="detection-item">
                    <div class="detection-dot" style="background:{hex_color};"></div>
                    <span class="detection-label">⚠️ {det['label']}</span>
                    <span class="detection-confidence">{det['confidence']:.0%}</span>
                </div>
                """, unsafe_allow_html=True)

        latency_ok = "✅" if r["meets_latency_target"] else "⚠️"
        st.markdown(f"""
        <div style="margin-top:1rem; font-size:0.75rem; color:#8899bb; text-align:right;">
            ⏱ Inferencia: {r['inference_ms']:.0f} ms {latency_ok} (meta LEAN: &lt;5000 ms) · 512×512
        </div>
        """, unsafe_allow_html=True)

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "usuario": current_user["display_name"],
            "modelo": "YOLOv11 + Mask R-CNN (mock-MVP)",
            "input_size": "512x512",
            "thresholds": {"confidence": confidence_threshold, "iou": iou_threshold},
            "resultados": {
                "total_piezas": r["total_teeth"],
                "piezas_con_hallazgos": r["affected_teeth"],
                "conteo_patologias": r["pathology_counts"],
                "severidad_global_pct": r["global_severity_pct"],
                "odontograma": r["odontogram"],
                "detecciones": r["detections"],
            },
        }
        # Encoder tolerante a tipos NumPy (np.float32, np.int64, np.ndarray).
        # json.dumps() solo serializa tipos nativos de Python; el modelo real
        # devolvera arrays de NumPy, asi que esto evita que el reporte falle.
        def _json_safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            raise TypeError(f"Tipo no serializable: {type(obj).__name__}")

        st.download_button(
            "📥 Descargar reporte JSON",
            data=json.dumps(report, indent=2, ensure_ascii=False, default=_json_safe),
            file_name=f"dental_report_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem; padding:1rem 1.5rem; border-top:1px solid #1e2d4a;
     display:flex; justify-content:space-between; align-items:center; font-size:0.72rem; color:#4a5c7a; flex-wrap:wrap; gap:0.5rem;">
    <span>⚠️ <strong style="color:#ffa502;">AVISO:</strong> Prototipo de investigación (MVP). No reemplaza diagnóstico profesional.</span>
    <span>DentalVision AI · MVP v0.2 · Edge AI · Streamlit</span>
</div>
""", unsafe_allow_html=True)
