"""
image_processor.py

Pre-procesamiento de radiografías dentales, alineado con la sección
"Implementación" del documento del proyecto:

  "Procesamiento Médico: Se utilizará OpenCV para la normalización,
   ecualización de histogramas y mejora de contraste de las imágenes
   radiográficas crudas, y PyDicom para la correcta manipulación de
   la metadata médica."

Pipeline: redimensionar con padding → CLAHE (OpenCV) → normalizar → tensor
"""

import numpy as np
from PIL import Image
import io

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class ImageProcessor:
    """
    Preprocesa imágenes radiográficas para el modelo de detección.

    Parámetros
    ----------
    target_size : tuple (width, height) — tamaño de entrada del modelo
    apply_clahe : bool — mejora de contraste adaptativo (recomendado para RX)
    clip_limit  : float — límite de recorte de CLAHE (2.0-3.0 es habitual en RX dental)
    """

    def __init__(self, target_size: tuple = (512, 512), apply_clahe: bool = True, clip_limit: float = 2.5):
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.clip_limit = clip_limit

        if not _HAS_CV2:
            print("[ImageProcessor] AVISO: OpenCV no disponible, usando fallback con PIL.")

    def preprocess(self, image: Image.Image) -> dict:
        """
        Ejecuta el pipeline completo de preprocesamiento.

        Returns
        -------
        dict con:
          tensor   : np.ndarray float32 (1, H, W, 3) — input para el modelo
          display  : PIL.Image — imagen preprocesada para mostrar al usuario
        """
        img = image.convert("RGB")
        img = self._resize_with_padding(img, self.target_size)

        if self.apply_clahe:
            img = self._apply_clahe_opencv(img) if _HAS_CV2 else self._apply_clahe_fallback(img)

        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = np.expand_dims(arr, axis=0)

        return {"tensor": tensor, "display": img}

    # ─── Resize con padding negro (preserva geometría dental) ────────────────
    @staticmethod
    def _resize_with_padding(image: Image.Image, target_size: tuple) -> Image.Image:
        tw, th = target_size
        iw, ih = image.size
        scale = min(tw / iw, th / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        resized = image.resize((nw, nh), Image.LANCZOS)
        padded = Image.new("RGB", (tw, th), (0, 0, 0))
        padded.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
        return padded

    # ─── CLAHE real vía OpenCV (espacio LAB, solo canal de luminancia) ───────
    def _apply_clahe_opencv(self, image: Image.Image) -> Image.Image:
        """
        Aplica CLAHE en el canal L del espacio LAB, evitando distorsión de
        color y maximizando el contraste de estructuras óseas/dentales.
        """
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)

        lab_eq = cv2.merge((l_eq, a, b))
        img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        img_rgb = cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img_rgb)

    # ─── Fallback sin OpenCV (equalización global simple con PIL) ────────────
    @staticmethod
    def _apply_clahe_fallback(image: Image.Image) -> Image.Image:
        from PIL import ImageOps
        return ImageOps.equalize(image)

    @staticmethod
    def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return buf.getvalue()

    @staticmethod
    def denoise(image: Image.Image) -> Image.Image:
        """Reducción de ruido opcional (útil en sensores RX de baja gama)."""
        if not _HAS_CV2:
            return image
        arr = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(arr, None, 6, 6, 7, 21)
        return Image.fromarray(denoised)
