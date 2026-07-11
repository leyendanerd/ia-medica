# 🦷 DentalVision AI — MVP

> Sistema de Inteligencia Artificial para diagnósticos dentales — ejecución **local (Edge AI)**
> Basado en el proyecto de innovación: *"Sistema de Inteligencia Artificial para diagnósticos dentales"* (UNIR — Maestría en IA)

---

## Contexto del proyecto

Este MVP corresponde a la fase **"Prototipar"** del Design Thinking descrito en el documento del proyecto. Implementa las historias de usuario definidas en el Sprint 1-3:

- ✅ **Historia 1** — Subir radiografía panorámica para análisis rápido
- ✅ **Historia 2** — Numeración automática de piezas (notación FDI universal)
- ✅ **Historia 3** — Mapa de calor sobre zonas con hallazgos clínicos
- ✅ **Historia 4** — Procesamiento 100% en red local, sin envío de datos a la nube

**Diferenciador frente a Overjet / Pearl:** sin dependencia de la nube → sin problemas de HIPAA/RGPD/LGPDPPSO, sin latencia de red, sin OPEX recurrente.

---

## Arquitectura del sistema

```
┌───────────────────────────────────────────────────────────────────┐
│                    CONTENEDOR DOCKER (Edge AI)                    │
│                                                                   │
│  ┌──────────────┐   ┌───────────────┐   ┌───────────────────┐   │
│  │  auth.py     │──▶│   Streamlit   │──▶│  ImageProcessor    │   │
│  │  (login      │   │   (UI Web)    │   │  • Resize+padding  │   │
│  │  local)      │   │  Puerto 8501  │   │  • CLAHE (OpenCV,  │   │
│  └──────────────┘   └───────┬───────┘   │    espacio LAB)    │   │
│                              │           └─────────┬──────────┘   │
│                              │                     │              │
│                    ┌─────────▼─────────┐          │              │
│                    │   DicomHandler    │          │              │
│                    │  (PyDicom)        │          │              │
│                    │  • Window C/W LUT │          │              │
│                    └─────────┬─────────┘          │              │
│                              │                     │              │
│                    ┌─────────▼─────────────────────▼──────────┐  │
│                    │       MockDentalInference                │  │
│                    │  (→ sustituir por YOLOv11 + Mask R-CNN)  │  │
│                    │  • Bounding boxes + numeración FDI (NMS) │  │
│                    │  • Segmentación: caries/hueso/quiste     │  │
│                    │  • Heatmap ponderado por hallazgo        │  │
│                    │  • Odontograma automático (32 piezas)    │  │
│                    └───────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

## Inicio rápido

### Con Docker (contenedor genérico x86_64)

```bash
docker compose up --build
open http://localhost:8501
```

**Credenciales de demostración:**
| Usuario | Contraseña | Rol |
|---|---|---|
| `odontologo` | `clinica2026` | Odontólogo |
| `admin` | `admin2026` | Administrador |

> ⚠️ Cambiar credenciales antes de cualquier despliegue real — ver `utils/auth.py`.

### Despliegue en NVIDIA Jetson Orin Nano (target final del proyecto)

```bash
docker build -f Dockerfile.jetson -t dental-vision:jetson .
docker run --runtime nvidia --network host dental-vision:jetson
```

Requiere JetPack + NVIDIA Container Toolkit configurados en el dispositivo. Ver comentarios en `Dockerfile.jetson` y `requirements.jetson.txt`.

### Desarrollo local sin Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

---

## Estructura del proyecto

```
dental-mvp/
├── app/
│   └── main.py                   # Interfaz Streamlit (con login)
├── utils/
│   ├── auth.py                   # Autenticación local básica
│   ├── image_processor.py        # CLAHE real (OpenCV, espacio LAB) + resize
│   ├── dicom_handler.py          # Lectura DICOM (PyDicom)
│   ├── mock_inference.py         # Motor YOLOv11+MaskRCNN (mock → real)
│   └── odontogram_renderer.py    # Odontograma SVG automático (FDI)
├── .streamlit/config.toml        # Tema oscuro + límites de subida
├── model/                        # ← Colocar pesos entrenados aquí
├── Dockerfile                    # Build x86_64 (desarrollo/testing)
├── Dockerfile.jetson             # Build ARM64 + TensorRT (producción)
├── requirements.txt
├── requirements.jetson.txt
└── docker-compose.yml
```

---

## Integrar los modelos reales

El motor `MockDentalInference` ya expone la API final (`predict()`, NMS, cálculo de severidad, odontograma). Para pasar a producción:

1. Entrenar **YOLOv11** (detección + numeración FDI) y **Mask R-CNN** (segmentación de caries/pérdida ósea/quistes) — dataset sugerido: Tufts Dental Database o DENTEX Challenge 2023.
2. Convertir a TensorFlow Lite (contenedor genérico) o exportar a TensorRT `.engine` (Jetson).
3. Colocar los pesos en `model/` y apuntar el constructor:
   ```python
   inference = MockDentalInference(model_path="model/dental_model.tflite")
   ```
4. El método `_run_tflite()` ya está implementado con el contrato de salida esperado (`boxes`, `scores`, `classes`).

---

## Clases clínicas detectadas

| Clase | Origen | Color anotación |
|---|---|---|
| Sano | YOLOv11 | Verde |
| Caries | Mask R-CNN | Rojo/coral |
| Pérdida ósea | Mask R-CNN | Ámbar |
| Posible quiste | Mask R-CNN | Púrpura |

---

## Métricas de validación (según sección 6 del documento)

- **Prueba de tiempo:** reducción ≥80% vs. odontograma manual (10-15 min → objetivo <3 min con revisión asistida).
- **Eficacia diagnóstica:** F1-Score >0.85 en detección de caries vs. panel de 3 odontólogos expertos (Gold Standard).
- **Latencia del sistema:** <5 segundos por imagen (criterio LEAN — el MVP expone este indicador en cada análisis).

---

## Seguridad y cumplimiento

- Procesamiento 100% on-premise — ninguna imagen sale de la red de la clínica.
- Autenticación básica local (usuario/contraseña con hash SHA-256 para el MVP; recomendado bcrypt/argon2 + gestión de usuarios cifrada en producción).
- Sin dependencias de servicios externos en tiempo de inferencia.

---

## ⚠️ Aviso legal

Prototipo de investigación (MVP académico). No ha sido validado clínicamente y **no debe utilizarse como sustituto del diagnóstico profesional odontológico**.
