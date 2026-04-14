import pickle
import shap
from MLPipeline import Node
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_file(filename):
    with open(filename, "rb") as f:
        out = pickle.load(f)
    return out

class shap_values(Node):
    def __init__(self, model_file, name="Shap values", explainer="TreeExplainer"):
        super().__init__(name)
        self.model_file = model_file
        self.explainer = explainer

        def cal_shap_values(data, filename):
            X, y = data
            model = load_file(self.model_file)

            # shap expaliner
            explainer = getattr(shap, self.explainer)(model)
            values = explainer.shap_values(X)
            with open(filename+"_shap_values.pkl", "wb") as f:
                pickle.dump(values, f)
            return values

        self.steps = [cal_shap_values]

#
