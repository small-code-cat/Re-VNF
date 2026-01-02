CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters /data/user0/PycharmProjects/MM-R5-main/output_v5/v0-20251226-065115/checkpoint-3044 \
    --merge_lora true \
    --output_dir paged_merged_qwen2_5_vl_sft_v5 \
    --safe_serialization true
