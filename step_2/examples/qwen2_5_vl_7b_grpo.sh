#!/bin/bash

set -x

MODEL_PATH=/data/user0/PycharmProjects/MM-R5-main/paged_merged_qwen2_5_vl_sft_v5  # replace it with your local file path

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/data/user0/PycharmProjects/EasyR1-main/page_noise_judge_output/v6/train.parquet \
    data.val_files=/data/user0/PycharmProjects/EasyR1-main/page_noise_judge_output/v6/test.parquet \
    data.image_dir=/data/user0/PycharmProjects \
    data.format_prompt=./examples/format_prompt/noise_judge.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/noise_judge.py:compute_score \
    worker.rollout.tensor_parallel_size=4 \
    trainer.experiment_name=qwen2_5_vl_7b_noise_judge_grpo_student_teacher \
    trainer.n_gpus_per_node=4