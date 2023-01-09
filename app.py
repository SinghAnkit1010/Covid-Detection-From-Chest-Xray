import tensorflow as tf
import numpy as np
from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = os.path.join('statics', 'uploads')

base_model = tf.keras.applications.vgg16.VGG16(include_top = False)

input = tf.keras.Input(shape = (224,224,3))
x = tf.keras.layers.Rescaling(scale=1./255)(input)
x = base_model(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units = 4096,activation = "relu")(x)
x = tf.keras.layers.Dense(units = 4096,activation = "relu")(x)
output = tf.keras.layers.Dense(units = 3,activation = "softmax")(x)
model = tf.keras.models.Model(inputs = input,outputs = output)
model.load_weights("./Covid_model.h5")

class covid_detect:
  def data_preprocessing(self,image):
    img = np.array(image)
    img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
    img = tf.image.resize(img,[224,224])
    return img
  
  def result(self,image):
    img = self.data_preprocessing(image)
    index = np.argmax(model.predict(img))
    if(index == 0):
      return "covid"
    if(index == 1):
      return "normal"
    if(index == 2):
      return "viral Pneumonia"
obj = covid_detect()

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
  return render_template("index.html")


@app.route("/predict", methods = ["GET","POST"])
def predict():
  if request.method == 'POST':
    uploaded_img = request.files['image']
    img_filename = secure_filename(uploaded_img.filename)
    uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
    img_file_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
    img = tf.keras.utils.load_img(img_file_path)
    prediction = obj.result(img)
    return render_template("index.html",prediction_result = prediction)
    

if __name__ == "__main__":
  app.run(debug=True)
