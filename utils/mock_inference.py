"""
mock_inference.py

Motor de inferencia del MVP — simula la salida combinada de:
  - YOLOv11  → detección rápida (bounding boxes) y numeración FDI de piezas
  - Mask R-CNN → segmentación de hallazgos: caries, pérdida ósea, quistes

Referencia: sección "Implementación" del documento del proyecto.
Cuando existan los pesos entrenados (.tflite / .engine para TensorRT),
sustituir _mock_detections() por _run_tflite() (ya incluido, listo para
apuntar a un modelo real con las mismas salidas estándar de detección).

Salida adicional: un odontograma automático (32 piezas, notación FDI),
que es uno de los objetivos específicos del proyecto ("automatizar la
generación de odontogramas").
"""

import numpy as np
import time
import random
from PIL import Image, ImageDraw, ImageFont


# Notación FDI (universal): cuadrantes 1-4, piezas 1-8 cada uno
FDI_QUADRANTS = {
    "upper_right": ["18", "17", "16", "15", "14", "13", "12", "11"],
    "upper_left":  ["21", "22", "23", "24", "25", "26", "27", "28"],
    "lower_left":  ["31", "32", "33", "34", "35", "36", "37", "38"],
    "lower_right": ["48", "47", "46", "45", "44", "43", "42", "41"],
}
ALL_FDI_TEETH = (
    FDI_QUADRANTS["upper_right"][::-1] + FDI_QUADRANTS["upper_left"]
    + FDI_QUADRANTS["lower_left"] + FDI_QUADRANTS["lower_right"][::-1]
)

# Clases clínicas detectables (Mask R-CNN) + color de anotación
PATHOLOGY_CLASSES = {
    "sano":         {"label": "Sano",             "color": (46,  213, 115, 190), "severity_weight": 0.0},
    "caries":       {"label": "Caries",           "color": (255, 71,  87,  210), "severity_weight": 1.0},
    "perdida_osea": {"label": "Pérdida ósea",     "color": (255, 165, 2,   210), "severity_weight": 0.8},
    "quiste":       {"label": "Posible quiste",   "color": (162, 89,  255, 210), "severity_weight": 1.2},
}


class MockDentalInference:
    """
    Motor de inferencia que simula la salida combinada YOLOv11 + Mask R-CNN
    hasta contar con los pesos entrenados reales. Misma API que la versión final.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.interpreter = None
        self.mock_mode = True
        if model_path:
            self._load_tflite(model_path)

    # ─── Carga del modelo real (TFLite / TensorRT en Jetson) ─────────────────
    def _load_tflite(self, path: str):
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.mock_mode = False
            print(f"[DentalInference] Modelo cargado: {path}")
        except Exception as e:
            print(f"[DentalInference] Error cargando modelo, usando mock: {e}")
            self.mock_mode = True

    # ─── Inferencia principal ─────────────────────────────────────────────────
    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        show_heatmap: bool = True,
        show_boxes: bool = True,
        show_labels: bool = True,
        heatmap_alpha: float = 0.45,
    ) -> dict:
        t0 = time.perf_counter()
        w, h = image.size

        if self.mock_mode:
            raw = self._mock_detections(w, h)
        else:
            raw = self._run_tflite(image)

        boxes, scores, classes, tooth_ids = self._nms(*raw, iou_threshold)

        detections = []
        for box, score, cls, tooth_id in zip(boxes, scores, classes, tooth_ids):
            if score >= confidence_threshold:
                meta = PATHOLOGY_CLASSES[cls]
                detections.append({
                    # Convertir a floats nativos de Python: np.ndarray/np.float32
                    # no son serializables por json.dumps() al exportar el reporte.
                    "box": [float(v) for v in box],
                    "confidence": float(score), "class": cls,
                    "label": f"Diente {tooth_id} — {meta['label']}",
                    "tooth_id": tooth_id,
                })

        detections.sort(key=lambda d: (d["class"] == "sano", -d["confidence"]))

        # Odontograma automático: una entrada por cada una de las 32 piezas
        odontogram = self._build_odontogram(detections)

        # Métricas globales
        total_teeth = len(odontogram)
        pathology_counts = {"caries": 0, "perdida_osea": 0, "quiste": 0}
        for d in detections:
            if d["class"] in pathology_counts:
                pathology_counts[d["class"]] += 1
        affected = sum(pathology_counts.values())
        severity_pct = self._compute_severity(detections, total_teeth)

        annotated = self._annotate(
            image.copy(), detections, w, h,
            show_heatmap=show_heatmap, show_boxes=show_boxes,
            show_labels=show_labels, heatmap_alpha=heatmap_alpha,
        )

        inference_ms = (time.perf_counter() - t0) * 1000

        return {
            "annotated_image": annotated,
            "detections": detections,
            "odontogram": odontogram,
            "total_teeth": total_teeth,
            "pathology_counts": pathology_counts,
            "affected_teeth": affected,
            "global_severity_pct": severity_pct,
            "inference_ms": inference_ms,
            "meets_latency_target": inference_ms < 5000,  # criterio LEAN del documento (<5s)
        }

    # ─── Mock: genera detecciones sintéticas plausibles ──────────────────────
    def _mock_detections(self, w: int, h: int):
        rng = random.Random(7)
        boxes, scores, classes, tooth_ids = [], [], [], []

        rows = {"upper": 0.15, "lower": 0.60}
        per_row = 16
        tooth_w = 1.0 / (per_row + 1)
        tooth_h = 0.18

        upper_ids = FDI_QUADRANTS["upper_right"][::-1] + FDI_QUADRANTS["upper_left"]
        lower_ids = FDI_QUADRANTS["lower_left"] + FDI_QUADRANTS["lower_right"][::-1]

        for row_name, ids in (("upper", upper_ids), ("lower", lower_ids)):
            row_y = rows[row_name]
            for k, tooth_id in enumerate(ids):
                x1 = (k + 0.5) * tooth_w - tooth_w * 0.42
                y1 = row_y
                x2 = x1 + tooth_w * rng.uniform(0.75, 1.0)
                y2 = y1 + tooth_h * rng.uniform(0.85, 1.15)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(1, x2), min(1, y2)

                roll = rng.random()
                if roll < 0.22:
                    cls = "caries"
                elif roll < 0.32:
                    cls = "perdida_osea"
                elif roll < 0.36:
                    cls = "quiste"
                else:
                    cls = "sano"

                score = rng.uniform(0.55, 0.97) if cls != "sano" else rng.uniform(0.80, 0.99)

                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                classes.append(cls)
                tooth_ids.append(tooth_id)

        return (
            np.array(boxes, dtype=np.float32),
            np.array(scores, dtype=np.float32),
            classes,
            tooth_ids,
        )

    # ─── Inferencia con modelo real (TFLite/TensorRT) ────────────────────────
    def _run_tflite(self, image: Image.Image):
        """
        Espera salidas estándar de detección:
          boxes [N,4], scores [N], classes [N] (0=sano,1=caries,2=perdida_osea,3=quiste)
        La numeración FDI se resuelve por posición geométrica relativa al arco dental
        (post-procesamiento geométrico, no lo entrega la red directamente).
        """
        import tensorflow as tf
        img_resized = image.resize((512, 512))
        input_data = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        cls_ids = self.interpreter.get_tensor(self.output_details[2]["index"])[0]

        id_to_class = {0: "sano", 1: "caries", 2: "perdida_osea", 3: "quiste"}
        classes = [id_to_class.get(int(c), "sano") for c in cls_ids]

        # Placeholder de asignación FDI por posición X (izq→der) — en producción
        # se resuelve con un módulo de tracking de arco dental.
        order = np.argsort(boxes[:, 0])
        tooth_ids = [ALL_FDI_TEETH[i % len(ALL_FDI_TEETH)] for i in range(len(order))]

        return boxes, scores, classes, tooth_ids

    # ─── NMS ──────────────────────────────────────────────────────────────────
    def _nms(self, boxes, scores, classes, tooth_ids, iou_threshold):
        if len(boxes) == 0:
            return boxes, scores, classes, tooth_ids

        order = np.argsort(scores)[::-1]
        keep = []
        order = list(order)
        while order:
            i = order.pop(0)
            keep.append(i)
            order = [j for j in order if self._iou(boxes[i], boxes[j]) < iou_threshold]

        keep = np.array(keep)
        return (
            boxes[keep], scores[keep],
            [classes[k] for k in keep],
            [tooth_ids[k] for k in keep],
        )

    @staticmethod
    def _iou(box_a, box_b):
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    # ─── Odontograma automático (objetivo del proyecto) ──────────────────────
    @staticmethod
    def _build_odontogram(detections: list) -> list:
        """
        Genera las 32 entradas del odontograma en notación FDI.
        Si una pieza no fue detectada, se marca como 'ausente'.
        """
        by_tooth = {d["tooth_id"]: d for d in detections}
        odontogram = []
        for tooth_id in ALL_FDI_TEETH:
            det = by_tooth.get(tooth_id)
            odontogram.append({
                "tooth_id": tooth_id,
                "status": det["class"] if det else "ausente",
                "confidence": det["confidence"] if det else None,
            })
        return odontogram

    # ─── Severidad global ponderada por tipo de hallazgo ─────────────────────
    @staticmethod
    def _compute_severity(detections: list, total_teeth: int) -> float:
        if total_teeth == 0:
            return 0.0
        weighted_sum = sum(
            PATHOLOGY_CLASSES[d["class"]]["severity_weight"] * d["confidence"]
            for d in detections if d["class"] != "sano"
        )
        max_possible = total_teeth * 1.2  # peor caso: todas quiste (peso máx 1.2)
        return float(min(100, (weighted_sum / max_possible) * 100))

    # ─── Renderizado ───────────────────────────────────────────────────────────
    def _annotate(self, image, detections, w, h, show_heatmap, show_boxes, show_labels, heatmap_alpha):
        image = image.convert("RGBA")

        if show_heatmap:
            heatmap = self._generate_heatmap(detections, w, h, heatmap_alpha)
            image = Image.alpha_composite(image, heatmap)

        if show_boxes or show_labels:
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for det in detections:
                if det["class"] == "sano" and not show_boxes:
                    continue
                x1, y1, x2, y2 = det["box"]
                px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                color = PATHOLOGY_CLASSES[det["class"]]["color"]
                border = (color[0], color[1], color[2], 220)

                if show_boxes:
                    thickness = 2 if det["class"] != "sano" else 1
                    for t in range(thickness):
                        draw.rectangle([px1 - t, py1 - t, px2 + t, py2 + t], outline=border)

                if show_labels:
                    label = f"{det['tooth_id']} {det['confidence']:.0%}"
                    font_size = max(11, min(15, w // 65))
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                    except Exception:
                        font = ImageFont.load_default()

                    bbox_txt = draw.textbbox((px1, py1 - font_size - 4), label, font=font)
                    draw.rectangle(
                        [bbox_txt[0] - 2, bbox_txt[1] - 2, bbox_txt[2] + 2, bbox_txt[3] + 2],
                        fill=(color[0], color[1], color[2], 215)
                    )
                    draw.text((px1, py1 - font_size - 4), label, fill=(255, 255, 255, 255), font=font)

            image = Image.alpha_composite(image, overlay)

        return image.convert("RGB")

    @staticmethod
    def _generate_heatmap(detections, w, h, alpha, max_dim: int = 256):
        """
        Genera el heatmap a BAJA RESOLUCION y luego lo escala al tamano final.

        Optimizacion de memoria: la version anterior creaba 6 arrays float32 del
        tamano completo de la imagen (np.mgrid + 4 canales). En una panoramica de
        2000x1000 eso son ~50 MB solo para el heatmap, recalculando la gaussiana
        para cada deteccion. Como un heatmap es difuso por naturaleza, generarlo
        a 256px y escalarlo es visualmente equivalente y usa ~60x menos memoria.
        """
        scale = min(max_dim / max(w, h), 1.0)
        hw, hh = max(1, int(w * scale)), max(1, int(h * scale))

        heat_r = np.zeros((hh, hw), dtype=np.float32)
        heat_g = np.zeros((hh, hw), dtype=np.float32)
        heat_b = np.zeros((hh, hw), dtype=np.float32)
        heat_a = np.zeros((hh, hw), dtype=np.float32)

        ys, xs = np.mgrid[0:hh, 0:hw].astype(np.float32)

        for det in detections:
            if det["class"] == "sano":
                continue
            color = PATHOLOGY_CLASSES[det["class"]]["color"]
            x1, y1, x2, y2 = det["box"]
            px1, py1, px2, py2 = int(x1 * hw), int(y1 * hh), int(x2 * hw), int(y2 * hh)
            cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
            rx, ry = max(1, (px2 - px1) / 2), max(1, (py2 - py1) / 2)

            gauss = np.exp(-((xs - cx) ** 2 / (2 * rx ** 2) + (ys - cy) ** 2 / (2 * ry ** 2)))
            weight = gauss * det["confidence"]

            heat_r += weight * color[0]
            heat_g += weight * color[1]
            heat_b += weight * color[2]
            heat_a = np.maximum(heat_a, weight)

        total_weight = np.maximum(heat_a, 1e-6)
        r = np.clip(heat_r / total_weight, 0, 255).astype(np.uint8)
        g = np.clip(heat_g / total_weight, 0, 255).astype(np.uint8)
        b = np.clip(heat_b / total_weight, 0, 255).astype(np.uint8)
        a_ch = np.clip(heat_a * alpha * 255, 0, 255).astype(np.uint8)

        small = Image.fromarray(np.stack([r, g, b, a_ch], axis=-1), mode="RGBA")
        # Escalar al tamano real de la imagen (bilineal = suave, propio de un heatmap)
        return small.resize((w, h), Image.BILINEAR) if (hw, hh) != (w, h) else small
