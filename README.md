# AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction
This repo is the Pytorch implementation of our manuscript titled [AirPhyNet: Physics-Guided Neural Networks for Air Quality Prediction]([https://github.com/Echo-Ji/STDEN/tree/main](https://arxiv.org/abs/2402.03784)). In this study, we present a novel physics guided differential equation network for precise air quality prediction over the next 72 hours with a physical meaning. The foundational training framework for this project is derived from [Echo-Ji](https://github.com/Echo-Ji/STDEN/tree/main).

## Framework
![AirPhyNet framework](https://github.com/kethmih/AirPhyNet/blob/main/img/AirPhyNet_Framework.png)

## Requirement
* scipy>=1.5.2
* numpy>=1.19.1
* pandas>=1.1.5
* pyyaml>=5.3.1
* pytorch>=1.7.1
* future>=0.18.2
* torchdiffeq>=0.2.0

Dependency can be installed using the following command:

```
pip install -r requirements.txt
```
## Data Preparation
The sample air quality data files for Beijing area are available at [Google Drive](https://drive.google.com/drive/folders/1RWOA7kaPFAgjedoszOLCAjoQR1XSv8Dt) and should be put into the `data/` folder for running the code.

## Model Traning and Evaluation
The following code can be used to train and evaluate the model. 
```bash
python main.py --config_filename = configs.yaml
```
## Citation
If you find our work useful in your research, please cite:
```tex
@article{hettige2024airphynet,
  title={AirPhyNet: Harnessing Physics-Guided Neural Networks for Air Quality Prediction},
  author={Hettige, Kethmi Hirushini and Ji, Jiahao and Xiang, Shili and Long, Cheng and Cong, Gao and Wang, Jingyuan},
  journal={arXiv preprint arXiv:2402.03784},
  year={2024}
}
```
