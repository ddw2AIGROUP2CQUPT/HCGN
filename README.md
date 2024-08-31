# [HCGN: Hierarchical Convolution and Graph Network for Predicting Knee Osteoarthritis](https://#)

Official PyTorch implementation of **HCGN**, from the following paper:


## Installation

Waiting ...

## Training & Evaluation

```
export OMP_NUM_THREADS=2
nohup python -m torch.distributed.run --nproc_per_node 2 --master_port 2488 ./train.py \
--model_name HCGN_CONVNEXT_TINY \
--img_size 224 \
--data_path PATH \
--do_online_eval \
--eval_freq 50 \
--weight_decay 0.001 \
--num_classes 5 \
--total_steps 500000 \
--batch_size 8 \
--cnn_lr 0.0001 \
--lr 0.001 \
--k 1 \
--heads 1 \
--optim sgd \
--smoothing True \
--patience 150 \
--pretrained \
--log_directory PATH \
--log_name NAME \
>> ./out/HCGN_CONVNEXT_TINY.txt &
```

## Citation
