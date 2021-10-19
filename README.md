# ms-convSTAR
Pytorch implementation for hierarchical time series classification with multi-stage convolutional RNN described in: 

[Crop mapping from image time series: deep learning with multi-scale label hierarchies. Turkoglu, Mehmet Ozgur and D'Aronco, Stefano and Perich, Gregor and Liebisch, Frank and Streit, Constantin and Schindler, Konrad and Wegner, Jan Dirk. Remote Sensing of Environment, 2021.](https://arxiv.org/pdf/2102.08820.pdf)


## [[Paper]](https://arxiv.org/pdf/2102.08820.pdf)  - [[Poster]](https://drive.google.com/file/d/1UkzijujTMTFv-fwTs4cFjFIRQlJQoUrq/view?usp=sharing)


<img src="https://github.com/0zgur0/ms-convSTAR/blob/master/imgs/model_drawing.png">


If you find our work useful in your research, please consider citing our paper:

```bash
@article{turkoglu2021msconvstar,
  title={Crop mapping from image time series: deep learning with multi-scale label hierarchies},
  author={Turkoglu, Mehmet Ozgur and D'Aronco, Stefano and Perich, Gregor and Liebisch, Frank and Streit, Constantin and Schindler, Konrad and Wegner, Jan Dirk},
  journal={Remote Sensing of Environment},
  volume={264},
  year={2021},
  publisher={Elsevier}
}
```


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




