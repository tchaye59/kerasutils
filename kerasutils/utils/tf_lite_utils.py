import tensorflow as tf

class LiteModel:

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, X, *args, **kwargs):
        return self.predict(X)

    def predict(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        if type(inp) not in [tuple, list]:
            inp = [inp, ]
        for input_det in self.input_details:
            input_index = input_det["index"]
            input_dtype = input_det["dtype"]
            x = inp[input_index].astype(input_dtype)
            self.interpreter.set_tensor(input_index, x)
        self.interpreter.invoke()
        result = []
        for output_det in self.output_details:
            output_index = output_det["index"]
            # output_dtype = output_det["dtype"]
            out = self.interpreter.get_tensor(output_index)
            result.append(out)
        if len(result) == 1:
            return result[0]
        return result


def to_TFLite(model):
    return LiteModel.from_keras_model(model)