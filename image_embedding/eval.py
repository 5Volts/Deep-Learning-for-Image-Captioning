import tensorflow as tf
import cv2
import numpy as np

class Transferred_Learning:
  def __init__(self):
    model_file = "image_embedding/mobilenet_v2_0.35_224_frozen.pb"

    input_layer = 'input'
    output_layer = "MobilenetV2/Predictions/Reshape_1"

    self.graph = self.load_graph(model_file)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    self.input_operation = self.graph.get_operation_by_name(input_name)
    self.output_operation = self.graph.get_operation_by_name(output_name)
    self.sess = tf.Session(graph=self.graph)

  def load_graph(self,model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph

  def read_tensor_from_img(self,img_np,
                            input_height=224,
                            input_width=224,
                            input_mean=0,
                            input_std=255):
    
    img_np = cv2.resize(img_np,(input_height,input_width))
    return np.divide(img_np, input_std)


  def get_result(self,img_np):
    t = self.read_tensor_from_img(img_np)
    results = self.sess.run(self.output_operation.outputs[0], {self.input_operation.outputs[0]: [t]})
    return results
