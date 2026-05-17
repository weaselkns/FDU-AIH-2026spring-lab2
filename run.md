# 默认已启用：英文小写、BIO 约束、标签加权、Cosine LR、按语言调参、按验证 F1 存最优权重
# 必须在项目根目录执行

cd /mnt/data/kw/kns/FDU-AIH-2026spring-lab2

# 推荐：英文单卡（lang_tune 默认 16 epoch）
CUDA_VISIBLE_DEVICES=3 python task3_transformer_crf/transformer_crf_ner.py \
  --lang English --batch-size 32 \
  --save-dir task3_transformer_crf/checkpoints

# 中文单卡（lang_tune 默认 12 epoch）
CUDA_VISIBLE_DEVICES=4 python task3_transformer_crf/transformer_crf_ner.py \
  --lang Chinese --batch-size 64 \
  --save-dir task3_transformer_crf/checkpoints

# 中英文顺序训练（单卡）
CUDA_VISIBLE_DEVICES=3 python task3_transformer_crf/transformer_crf_ner.py \
  --lang both --save-dir task3_transformer_crf/checkpoints

# 四卡 DDP（默认 --eval-every 500：24 epoch 时只在第 24 轮验证一次，最快）
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --standalone --nproc_per_node=4 \
  task3_transformer_crf/transformer_crf_ner.py \
  --lang both --batch-size 64 --num-workers 4 \
  --save-dir task3_transformer_crf/checkpoints

# 想中途盯 F1：--eval-every 2 或 5
# 完全不要验证、不要最优 checkpoint：--no-epoch-eval
