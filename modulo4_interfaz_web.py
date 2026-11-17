import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import cv2
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from modulo1_datos_modelo import MODEL_PATH, IMG_SIZE, CLASS_NAMES
from modulo3_gradcam import generar_mapa_calor

if not os.path.exists(MODEL_PATH):
  raise FileNotFoundError(
      f"No se encontró el modelo en '{MODEL_PATH}'.\n"
      "Ejecuta primero 'python modulo2_entrenar.py' para entrenarlo y"
      ' guardarlo.'
  )

model = load_model(MODEL_PATH)


def diagnosticar_radiografia(imagen_entrada):
  if imagen_entrada is None:
    return None, None

  if len(imagen_entrada.shape) == 2:
    original_rgb = cv2.cvtColor(imagen_entrada, cv2.COLOR_GRAY2RGB)
  else:
    original_rgb = imagen_entrada.copy()

  # Preprocesamiento idéntico al entrenamiento
  img_resized = cv2.resize(original_rgb, IMG_SIZE)
  img_normalized = img_resized.astype('float32') / 255.0
  img_batch = np.expand_dims(img_normalized, axis=0)

  # Predicción
  preds = model.predict(img_batch, verbose=0)[0]
  probabilidades = {
      CLASS_NAMES[0]: float(preds[0]),
      CLASS_NAMES[1]: float(preds[1]),
  }

  # Mapa de calor Grad-CAM
  mapa_calor = generar_mapa_calor(
      model=model,
      img_array=img_batch,
      original_img=original_rgb,
      last_conv_layer_name='block5_conv3',
      alpha=0.45,
  )

  return probabilidades, mapa_calor


with gr.Blocks(theme=gr.themes.Soft(), title='Diagnóstico de Neumonía') as app:
  gr.Markdown(
      '# Sistema de Apoyo Diagnóstico: Detección de Neumonía con Deep Learning'
  )
  gr.Markdown(
      'Cargue una radiografía de tórax. El sistema clasificará si existe'
      ' **Neumonía** o estado **Normal** y proyectará el **mapa de calor'
      ' Grad-CAM** indicando las zonas pulmonares de mayor activación.'
  )

  with gr.Row():
    with gr.Column():
      entrada_rx = gr.Image(
          label='Subir Radiografía de Tórax', type='numpy', height=380
      )
      boton_analizar = gr.Button('Analizar Radiografía', variant='primary')

    with gr.Column():
      salida_diagnostico = gr.Label(
          num_top_classes=2, label='Probabilidad Diagnóstica'
      )
      salida_calor = gr.Image(
          label='Mapa de Calor Grad-CAM (Explicabilidad)', height=380
      )

  boton_analizar.click(
      fn=diagnosticar_radiografia,
      inputs=entrada_rx,
      outputs=[salida_diagnostico, salida_calor],
  )

if __name__ == '__main__':
  app.launch(server_name='127.0.0.1', server_port=7860, inbrowser=True)