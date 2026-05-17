# Task3 默认与历史最佳验证 F1 对齐：12 epoch、每卡 batch 64、每 epoch 验证并保存最优权重
# 英文：128 维 / 3 层；中文：128 维 / 2 层；BIO 约束 + 小写 + 标签加权 + Cosine LR
# 必须在项目根目录执行

cd /mnt/data/kw/kns/FDU-AIH-2026spring-lab2

# 四卡 DDP（不传参即默认：epochs=12, batch-size=64, eval-every=1）
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --standalone --nproc_per_node=4 \
  pj2/part3/transformer_crf_ner.py \
  --lang both --batch-size 64 --num-workers 4 \
  --save-dir pj2/part3/checkpoints

# 单卡中英文
CUDA_VISIBLE_DEVICES=3 python pj2/part3/transformer_crf_ner.py \
  --lang both --save-dir pj2/part3/checkpoints

# 少做验证加快速度：--eval-every 3
# 完全不做验证（不推荐）：--no-epoch-eval

# Bonus：模板 CRF 中文 NER（CPU，约半分钟）
# python pj2/bonus/train_template_crf_chinese.py
