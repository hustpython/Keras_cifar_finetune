# 定义层
import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

# 狂阶图片指定尺寸
target_size = (229, 229) #fixed size for InceptionV3 architecture

# 预测函数
# 输入：model，图片，目标尺寸
# 输出：预测predict
def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]

# 画图函数
# 预测之后画图，这里默认是猫狗，当然可以修改label

labels = ("bus", "dinosaurs","elephants","flowers","horse")
def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')
  plt.figure()
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()

# 载入模型
model = load_model('fine-tuning.h5')

# 本地图片
image_local = 'data/train_sym/horse/720.jpg'
img = image.load_img(image_local)
preds = predict(model, img, target_size)
print(preds)
plot_preds(img, preds)

# 图片URL
#response = requests.get(image_url)
#img = Image.open(BytesIO(response.content))
#preds = predict(model, img, target_size)
#plot_preds(img, preds)
