from sklearn.datasets import load_iris, load_digits, load_diabetes, load_boston, load_wine, load_breast_cancer
import rps_ams_auto_modeling

x = load_wine().data
y = load_wine().target

score = rps_ams_auto_modeling.rps_ams_auto_modeling(x, y)
