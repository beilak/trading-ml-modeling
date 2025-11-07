# trading-ml-modeling


## Install

```bash
brew install libomp
brew install graphviz
poetry install
```


```
Open models > model > model_logistic_regression.ipynb

model with:
1 - No reg
2 - L1
3 - L2 
4 - elasticnet
```


Feature Visualisator

```bash
docker build -f Dockerfile.fn.visualisation.py  -t fe-feature_visualisation-app .
docker run -p 8501:8501 fe-feature_visualisation-app             
 ```