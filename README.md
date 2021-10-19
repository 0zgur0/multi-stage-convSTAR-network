# ms-convSTAR
Pytorch implementation for hierarchical time series classification with multi-stage convolutional RNN based on our RSE paper: Turkoglu, Mehmet Ozgur and D'Aronco, Stefano and Perich, Gregor and Liebisch, Frank and Streit, Constantin and Schindler, Konrad and Wegner, Jan Dirk: Crop mapping from image time series: deep learning with multi-scale label hierarchies. Remote Sensing of Environment, 2021


# [[Paper]](https://arxiv.org/pdf/2102.08820.pdf)  - [[Poster]](https://drive.google.com/file/d/1UkzijujTMTFv-fwTs4cFjFIRQlJQoUrq/view?usp=sharing)


<img src="https://github.com/0zgur0/ms-convSTAR/blob/master/imgs/model_drawing.png">


## ZueriCrop Dataset
Download the dataset via https://polybox.ethz.ch/index.php/s/uXfdr2AcXE3QNB6

## Getting Started

Train the model e.g., for fold:1 with 
```bash
python3 train.py --data /path/to/data --fold 1
```


Test the trained model e.g., for fold:1 with 
```bash
python3 test.py --data /path/to/data --fold 1 --snapshot /path/to/trained_model
```


## Citation
```bash
@inproceedings{turkoglu2021visual,
  title={Visual Camera Re-Localization Using Graph Neural Networks and Relative Pose Supervision},
  author={Turkoglu, Mehmet Ozgur and Brachmann, Eric and Schindler, Konrad and Brostow, Gabriel and Monszpart, Aron},
  booktitle={2019 International Conference on 3D Vision (3DV)},
  year={2021},
  organization={IEEE}
}
}
```
