# TIPCB

### Original Program
Our program base on TIPCB. Please visit [Here](https://github.com/OrangeYHChen/TIPCB).

### Disclaimer
* This program was create by 4th undergraduate of Mahidol university
* This program was create for compair performance between Thai full text dataset and cut off Thai dataset

### Prerequisites
* Pytorch 1.1
* cuda 9.0
* python 3.6
* GPU Memory>=12G

### Datasets
We evaluate our method on CUHK-PEDES. Please visit [Here](http://xiaotong.me/static/projects/person-search-language/dataset.html). And our evaluate this dataset on Thai


### Usage
* 'Cutting word json.ipynb' using for cutting Thai full word on thai POS tagging
* 'npz_export.py' using for change json to npz format by follow command :
 
``
python npz_export.py --json_path=data/caption_all_thai2.json --npz_out_folder=data/BERT_encode/
``

* 'train.py' using for training program by follow command :

``
python train.py --max_length 64 --batch_size 64 --num_epoches 40 --adam_lr 0.003 --gpus 0 --dir ./data --embedding_type=WangchanBERTa
``

* 'test.py' using for testing program by follow command :

``
python test_model.py --dir=data --model_path=log/Experiment01 --checkpoint_dir=log/Experiment01 --log_test_dir=log/Experiment01 --gpus=0 --embedding_type=WangchanBERTa --num_epoches=40
``

### Evaluate

* Before cutting

| Top-1 | Top-5 | Top-10 |
| :------: | :------: | :------: |
| 47.94 | 69.63 | 78.61 |

* After cutting

| Top-1 | Top-5 | Top-10 |
| :------: | :------: | :------: |
| 46.88 | 68.90 | 77.09 |
