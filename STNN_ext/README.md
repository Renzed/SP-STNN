# (Selfless)-Prop-STNN
This is the PyTorch implementation of paper *["Space Meets Time: Local Spacetime Neural Network For Trafﬁc Flow Forecasting"](https://arxiv.org/pdf/2109.05225)*.
With the following changes:
- Distances are no longer calculated using exp, just raw distance divided by 10 (could be reverted). This requires the user to select a suitable scale.
- Vp selection now uses half near nodes as in the original STNN, and half physicallyy most distant at travel time of the prediction horizon
- Dynamic Vp selection based on current travel time
- L2 loss
## Installation
```
pip install -r requirements.txt
```

## Requirements
- pytorch (1.7 or later)
- numpy
- prettytable
- tqdm


## Train
STNN was originally trained in the following way:
```
# Train on PeMS-Bay
python train.py --data data/PeMS-Bay --t_history 12 --t_pred 12 --keep_ratio 0.2
```
However, I suggest simply editing the train.py file to reduce how much needs to be typed in the command line. This is especially useful when using many datasets.

## Test
STNN was tested in the following way:
```
python test.py --data data/METR-LA --model weights/STNN-combined.state.pt
python test.py --data data/PeMS-Bay --model weights/STNN-combined.state.pt
```
However, the `graphing.py` module generates a `pandas` dataframe for easy further handling of model predictions.

## Citation
I suggest referencing the original STNN paper, they did the grunt of the work 
```
@article{yang2021space,
  title={Space Meets Time: Local Spacetime Neural Network For Traffic Flow Forecasting},
  author={Yang, Song and Liu, Jiamou and Zhao, Kaiqi},
  journal={arXiv preprint arXiv:2109.05225},
  year={2021}
}
@Thesis{dijkhuizen2024traffic,
    title={Information flow based machine learning traffic prediction},
    author={Renze Dijkhuizen},
    type={mathesis},
    institution={Technische Universiteit Delft},
    year={2024}
}
```
