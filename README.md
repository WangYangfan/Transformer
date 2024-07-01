# Transformer
a simple implementation of Transformer for ChineseNMT

**Train**

```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
    --name example \
    --num_shards 100 \
    --save_log \
    --save_model
```

**Eval**

```bash
CUDA_VISIBLE_DEVICES=7 python eval.py \
    --model_path ./output/example_shards_None/checkpoint-110590 \
    --num_shards 100
```
