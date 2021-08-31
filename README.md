# ms-convSTAR
Pytorch code for hierarchical time series classification with multi-stage convolutional RNN based on our Remote Sensing of Environment, 2021 paper:
https://arxiv.org/pdf/2102.08820.pdf

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
@article{turkoglu2021gating,
  title={Crop mapping from image time series: deep learning with multi-scale label hierarchies},
  author={Turkoglu, Mehmet Ozgur and D'Aronco, Stefano and Perich, Gregor and Liebisch, Frank and Streit, Constantin and Schindler, Konrad and Wegner, Jan Dirk},
  journal={Remote Sensing of Environment},
  volume={264},
  year={2021},
  publisher={Elsevier}
}
```
