(东南大学计算机设计竞赛D01-013组代码)
### Introduction
This is a pytorch implemention of the encoder-forecaster model for precipitation nowcasting. The code is based on [Precipitation-Nowcasting](https://github.com/Hzzone/Precipitation-Nowcasting). For more information about TrajGRU, please refer to [HKO-7](https://github.com/sxjscience/HKO-7).


### Use
Train the model: Firstly you should apply for HKO-7 Dataset from [HKO-7](https://github.com/sxjscience/HKO-7), and modify config.py to find the dataset path. Secondly and last, run `python3 trajmase.py`. After training, you can run 'python3 pred.py' to get the prediction results.
Generate a new dataset configuration file by excluding days with rainfall intensity lower than a certain value: python pkl.py. You may need to modify the code to find the right path.
### Environment
Python 3.6+, PyTorch 1.1.0 and Ubuntu.

### Citation

```
@inproceedings{xingjian2017deep,
    title={Deep learning for precipitation nowcasting: a benchmark and a new model},
    author={Shi, Xingjian and Gao, Zhihan and Lausen, Leonard and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-kin and Woo, Wang-chun},
    booktitle={Advances in Neural Information Processing Systems},
    year={2017}
}
@inproceedings{xingjian2015convolutional,
  title={Convolutional LSTM network: A machine learning approach for precipitation nowcasting},
  author={Xingjian, SHI and Chen, Zhourong and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-Kin and Woo, Wang-chun},
  booktitle={Advances in neural information processing systems},
  pages={802--810},
  year={2015}
}
```

