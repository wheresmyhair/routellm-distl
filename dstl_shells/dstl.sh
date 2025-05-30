all_tasks=bbh,truthfulqa,agieval,mmlu,ifeval,arc_easy,arc_challenge,hellaswag,openbookqa,piqa,social_iqa,winogrande,wikitext,gpqa,minerva_math,gsm8k,humaneval,mbpp

HF_ALLOW_CODE_EVAL="1" CUDA_VISIBLE_DEVICES=0 nohup lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,port=12330,gpu_memory_utilization=0.8 \
    --tasks $all_tasks \
    --batch_size 1 \
    --cache_requests refresh \
    --num_fewshot 0 \
    --output_path ./allres \
    --log_samples \
    --confirm_run_unsafe_code > ./allres/llama3.2-3b.log 2>&1 &

# HF_ALLOW_CODE_EVAL="1" CUDA_VISIBLE_DEVICES=1 nohup lm_eval --model sglang \
#     --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,dtype=auto,port=12331 \
#     --tasks $all_tasks \
#     --batch_size 1 \
#     --cache_requests refresh \
#     --num_fewshot 0 \
#     --output_path ./allres \
#     --log_samples \
#     --confirm_run_unsafe_code > ./allres/llama3.2-1b.log 2>&1 &

# HF_ALLOW_CODE_EVAL="1" CUDA_VISIBLE_DEVICES=2 nohup lm_eval --model sglang \
#     --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=auto,port=12332 \
#     --tasks $all_tasks \
#     --batch_size 1 \
#     --cache_requests refresh \
#     --num_fewshot 0 \
#     --output_path ./allres \
#     --log_samples \
#     --confirm_run_unsafe_code > ./allres/llama3.1-8b.log 2>&1 &

# HF_ALLOW_CODE_EVAL="1" CUDA_VISIBLE_DEVICES=3 nohup lm_eval --model sglang \
#     --model_args pretrained=google/gemma-3-1b-it,dtype=auto,port=12333 \
#     --tasks $all_tasks \
#     --batch_size 1 \
#     --cache_requests refresh \
#     --num_fewshot 0 \
#     --output_path ./allres \
#     --log_samples \
#     --confirm_run_unsafe_code > ./allres/gemma-3-1b.log 2>&1 &

# HF_ALLOW_CODE_EVAL="1" CUDA_VISIBLE_DEVICES=4 nohup lm_eval --model sglang \
#     --model_args pretrained=google/gemma-3-4b-it,dtype=auto,port=12334 \
#     --tasks $all_tasks \
#     --batch_size 1 \
#     --cache_requests refresh \
#     --num_fewshot 0 \
#     --output_path ./allres \
#     --log_samples \
#     --confirm_run_unsafe_code > ./allres/gemma-3-4b.log 2>&1 &




