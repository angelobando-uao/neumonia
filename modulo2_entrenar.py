import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

from modulo1_datos_modelo import (
    IMG_SIZE,
    MODEL_PATH,
    construir_modelo,
    preparar_dataset,
)
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def entrenar():
  train_dir, test_dir = preparar_dataset()

  train_datagen = ImageDataGenerator(
      rescale=1.0 / 255,
      shear_range=0.15,
      zoom_range=0.15,
      rotation_range=10,
      horizontal_flip=True,
  )
  test_datagen = ImageDataGenerator(rescale=1.0 / 255)

  training_set = train_datagen.flow_from_directory(
      train_dir, target_size=IMG_SIZE, batch_size=16, class_mode='categorical'
  )

  test_set = test_datagen.flow_from_directory(
      test_dir,
      target_size=IMG_SIZE,
      batch_size=16,
      class_mode='categorical',
      shuffle=False,
  )

  # Balanceo automático de clases (NORMAL vs PNEUMONIA)
  clases = training_set.classes
  pesos_valores = compute_class_weight(
      class_weight='balanced', classes=np.unique(clases), y=clases
  )
  class_weights = dict(enumerate(pesos_valores))
  print(f'Mapeo de clases: {training_set.class_indices}')

  model = construir_modelo()

  callbacks = [
      ModelCheckpoint(
          MODEL_PATH,
          monitor='val_accuracy',
          save_best_only=True,
          mode='max',
          verbose=1,
      ),
      EarlyStopping(
          monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
      ),
  ]

  model.fit(
      training_set,
      validation_data=test_set,
      epochs=10,
      class_weight=class_weights,
      callbacks=callbacks,
  )
  print(f'\nEntrenamiento finalizado. Modelo guardado en:\n{MODEL_PATH}')


if __name__ == '__main__':
  entrenar()