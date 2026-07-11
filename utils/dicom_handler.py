"""
dicom_handler.py

Carga y convierte archivos DICOM de radiografías dentales a PIL.Image,
usando PyDicom — tal como especifica el documento del proyecto para
"la correcta manipulación de la metadata médica".
"""

from PIL import Image
import numpy as np
import io


class DicomHandler:
    """Convierte bytes DICOM → PIL.Image RGB, con extracción de metadata clínica."""

    def load(self, file_bytes: bytes) -> Image.Image:
        try:
            import pydicom
            from pydicom.filebase import DicomBytesIO

            ds = pydicom.dcmread(DicomBytesIO(file_bytes))
            return self._dicom_to_pil(ds)

        except ImportError:
            try:
                return Image.open(io.BytesIO(file_bytes)).convert("RGB")
            except Exception:
                raise RuntimeError("No se pudo abrir el archivo. Instala pydicom: pip install pydicom")
        except Exception as e:
            raise RuntimeError(f"Error leyendo DICOM: {e}")

    @staticmethod
    def _dicom_to_pil(ds) -> Image.Image:
        pixel_array = ds.pixel_array.astype(np.float32)

        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = float(ds.WindowCenter) if not hasattr(ds.WindowCenter, "__len__") else float(ds.WindowCenter[0])
            ww = float(ds.WindowWidth) if not hasattr(ds.WindowWidth, "__len__") else float(ds.WindowWidth[0])
            lower, upper = wc - ww / 2, wc + ww / 2
            pixel_array = np.clip(pixel_array, lower, upper)
        else:
            lower, upper = pixel_array.min(), pixel_array.max()

        p_range = upper - lower
        pixel_array = (pixel_array - lower) / p_range * 255.0 if p_range > 0 else np.zeros_like(pixel_array)
        pixel_array = pixel_array.clip(0, 255).astype(np.uint8)

        if pixel_array.ndim == 2:
            img = Image.fromarray(pixel_array, mode="L").convert("RGB")
        elif pixel_array.ndim == 3:
            if pixel_array.shape[-1] == 3:
                img = Image.fromarray(pixel_array, mode="RGB")
            elif pixel_array.shape[-1] == 4:
                img = Image.fromarray(pixel_array, mode="RGBA").convert("RGB")
            else:
                img = Image.fromarray(pixel_array[0], mode="L").convert("RGB")
        else:
            raise ValueError(f"Forma de pixel no soportada: {pixel_array.shape}")

        return img

    @staticmethod
    def extract_metadata(file_bytes: bytes) -> dict:
        try:
            import pydicom
            from pydicom.filebase import DicomBytesIO

            ds = pydicom.dcmread(DicomBytesIO(file_bytes), stop_before_pixels=True)
            return {
                "patient_id":   getattr(ds, "PatientID", "N/D"),
                "study_date":   getattr(ds, "StudyDate", "N/D"),
                "modality":     getattr(ds, "Modality", "N/D"),
                "manufacturer": getattr(ds, "Manufacturer", "N/D"),
                "rows":         getattr(ds, "Rows", None),
                "columns":      getattr(ds, "Columns", None),
                "bits_stored":  getattr(ds, "BitsStored", None),
            }
        except Exception:
            return {}
