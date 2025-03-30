lm_eval --model hf \
        --model_args pretrained=gpt2 \
        --tasks mmlu \
        --device cuda:0 \
        --output_path lm-eval-gpt2-mmlu