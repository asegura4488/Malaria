import os
import random
import shutil
from pathlib import Path

# ============ Configuración ============
sanos_dir = Path("sanos")        # carpeta con imágenes sanas
nosanos_dir = Path("nosanos")    # carpeta con imágenes no sanas
output_dir = Path("sampled")     # carpeta de salida
n_total = 250                    # número total de pares a elegir
# =======================================

# 1) Encontrar los pares (imagen principal y su _ref)
def find_pairs(folder: Path):
    imgs = [p for p in folder.glob("*.png") if not p.name.endswith("_ref.png")]
    pairs = []
    for img in imgs:
        ref = folder / f"{img.stem}_ref.png"
        if ref.exists():
            pairs.append((img, ref))
    return pairs

sanos_pairs = find_pairs(sanos_dir)
nosanos_pairs = find_pairs(nosanos_dir)

print(f"Sanos: {len(sanos_pairs)} pares encontrados")
print(f"No sanos: {len(nosanos_pairs)} pares encontrados")

# 2) Seleccionar de forma aleatoria
n_each = n_total // 2  # balanceado
random.seed(42)        # para reproducibilidad
sanos_sample = random.sample(sanos_pairs, min(n_each, len(sanos_pairs)))
nosanos_sample = random.sample(nosanos_pairs, min(n_each, len(nosanos_pairs)))

# 3) Crear carpeta de salida
if output_dir.exists():
    shutil.rmtree(output_dir)
(output_dir / "sanos").mkdir(parents=True, exist_ok=True)
(output_dir / "nosanos").mkdir(parents=True, exist_ok=True)

# 4) Copiar archivos seleccionados
def copy_pairs(pairs, outfolder):
    for img, ref in pairs:
        shutil.copy(img, outfolder / img.name)
        shutil.copy(ref, outfolder / ref.name)

copy_pairs(sanos_sample, output_dir / "sanos")
copy_pairs(nosanos_sample, output_dir / "nosanos")

print(f"Se copiaron {len(sanos_sample)} pares en sanos y {len(nosanos_sample)} pares en nosanos.")
print(f"Carpeta final: {output_dir}")

