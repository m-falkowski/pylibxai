#!/usr/bin/env bash

GREEN="\033[32m"
CLR="\033[0m"

# Run unit tests for pylibxai
python -m pytest -v pylibxai/pylibxai_context/test_pylibxai_context.py \
                    pylibxai/Interfaces/test_interfaces.py \
                    pylibxai/Explainers/test_explainers.py \
                    pylibxai/Views/test_web_view.py


echo -e "${GREEN}[TEST1]${CLR} CNN14, LIME, SHAP, Sandman 5s"
mkdir -p ./cnn14_expl/ &&
python ./pylibxai/pylibxai_explain.py -w ./cnn14_expl/ -m CNN14 --explainer=lime,shap --target=0 -i ./data/sandman_5s.wav &&
rm -rf ./cnn14_expl/

echo -e "${GREEN}[TEST2]${CLR} HarmonicCNN, LIME, SHAP, Sandman 5s"
mkdir -p ./harmoniccnn_expl/ &&
python ./pylibxai/pylibxai_explain.py -w ./harmoniccnn_expl/ -m HarmonicCNN --explainer=lime,shap --target=0 \
                           -i ./data/sandman_5s.wav &&
rm -rf ./harmoniccnn_expl/

echo -e "${GREEN}[TEST3]${CLR} GtzanCNN, SHAP, LRP"
mkdir -p ./gtzancnn_expl/ &&
python ./pylibxai/pylibxai_explain.py -w ./gtzancnn_expl/ -m GtzanCNN --explainer=shap,lrp --target=jazz \
                           -i ./data/gtzan_jazz.wav &&
rm -rf ./gtzancnn_expl/
