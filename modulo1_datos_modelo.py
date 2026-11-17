import os
import zipfile
import tensorflow as tf
print(tf.__version__)

# Silenciar avisos informativos de TensorFlow y mantener aceleración oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Rutas dinámicas basadas en la ubicación de este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# Posibles ubicaciones de tu archivo ZIP en VS Code
ZIP_CANDIDATES = [
    os.path.join(BASE_DIR, 'Pneumonia Detection Dataset.zip'),
]

EXTRACT_DIR = os.path.join(BASE_DIR, 'dataset_extraido')
MODEL_PATH = os.path.join(BASE_DIR, 'best_pneumonia_vgg16.keras')
IMG_SIZE = (224, 224)
CLASS_NAMES = ['NORMAL', 'NEUMONÍA']


def preparar_dataset():
  """Busca y descomprime el ZIP del dataset, devolviendo las rutas de train y test."""
  zip_encontrado = None
  for ruta in ZIP_CANDIDATES:
    if os.path.exists(ruta):
      zip_encontrado = ruta
      break

  if zip_encontrado is None:
    raise FileNotFoundError(
        "No se encontró 'pneumonia detection datase.zip'.\n"
        f"Colócalo dentro de '{BASE_DIR}' o '{PARENT_DIR}'."
    )

  if not os.path.exists(EXTRACT_DIR):
    print(f'Extrayendo {zip_encontrado}...')
    with zipfile.ZipFile(zip_encontrado, 'r') as zip_ref:
      zip_ref.extractall(EXTRACT_DIR)
    print('Extracción completada.')
  else:
    print(f'Dataset ya extraído en: {EXTRACT_DIR}')

  # Localizar automáticamente las subcarpetas 'train' y 'test' sin importar cómo venga comprimido
  train_dir, test_dir = None, None
  for root, dirs, _ in os.walk(EXTRACT_DIR):
    if 'train' in dirs and 'test' in dirs:
      train_dir = os.path.join(root, 'train')
      test_dir = os.path.join(root, 'test')
      break

  if not train_dir or not test_dir:
    raise FileNotFoundError(
        "El ZIP se extrajo pero no se encontraron las carpetas 'train' y"
        " 'test'."
    )

  return train_dir, test_dir


def construir_modelo(learning_rate=1e-4):
  """Construye el modelo VGG16 con Transfer Learning."""
  base_model = VGG16(
      input_shape=(224, 224, 3), weights='imagenet', include_top=False
  )
  for layer in base_model.layers:
    layer.trainable = False

  x = GlobalAveragePooling2D(name='gap')(base_model.output)
  x = Dense(128, activation='relu', name='fc_1')(x)
  x = Dropout(0.5, name='dropout')(x)
  predictions = Dense(
      len(CLASS_NAMES), activation='softmax', name='predictions'
  )(x)

  model = Model(inputs=base_model.input, outputs=predictions)
  model.compile(
      optimizer=Adam(learning_rate=learning_rate),
      loss='categorical_crossentropy',
      metrics=['accuracy'],
  )
  return model


if __name__ == '__main__':
  # Prueba rápida de descompresión y verificación de arquitectura
  t_dir, v_dir = preparar_dataset()
  print(f'Ruta Train: {t_dir}')
  print(f'Ruta Test:  {v_dir}')
  modelo = construir_modelo()
  print('Arquitectura VGG16 construida correctamente.')