import cv2
import numpy as np
import tensorflow as tf


def generar_mapa_calor(
    model,
    img_array,
    original_img,
    last_conv_layer_name='block5_conv3',
    alpha=0.45,
):
  """
    Genera y superpone el mapa de calor Grad-CAM sobre la radiografía original.
    """
  grad_model = tf.keras.models.Model(
      inputs=model.inputs,
      outputs=[model.get_layer(last_conv_layer_name).output, model.output],
  )

  with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    pred_index = tf.argmax(predictions[0])
    class_channel = predictions[:, pred_index]

  grads = tape.gradient(class_channel, conv_outputs)
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  conv_outputs = conv_outputs[0]
  heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)

  heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
  heatmap = heatmap.numpy()

  heatmap_resized = cv2.resize(
      heatmap, (original_img.shape[1], original_img.shape[0])
  )
  heatmap_uint8 = np.uint8(255 * heatmap_resized)
  heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
  heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

  superimposed_img = cv2.addWeighted(
      original_img, 1 - alpha, heatmap_colored, alpha, 0
  )
  return superimposed_img