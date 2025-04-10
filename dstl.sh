all_tasks=bbh,truthfulqa,agieval,mmlu,ifeval,arc_easy,arc_challenge,hellaswag,openbookqa,piqa,social_iqa,winogrande,wikitext,gpqa,minerva_math,gsm8k_yaml,humaneval,mbpp

nohup CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,dtype=auto \
    --tasks $tasks \
    --batch_size auto \
    --cache_requests refresh \
    --num_fewshot 0 \
    --output_path ./allres \
    --log_samples > ./allres/llama3.2-3b.log 2>&1 &

nohup CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,dtype=auto \
    --tasks $tasks \
    --batch_size auto \
    --cache_requests refresh \
    --num_fewshot 0 \
    --output_path ./allres \
    --log_samples > ./allres/llama3.2-1b.log 2>&1 &

nohup CUDA_VISIBLE_DEVICES=2 lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=auto \
    --tasks $tasks \
    --batch_size auto \
    --cache_requests refresh \
    --num_fewshot 0 \
    --output_path ./allres \
    --log_samples > ./allres/llama3.1-8b.log 2>&1 &

nohup CUDA_VISIBLE_DEVICES=3 lm_eval --model vllm \
    --model_args pretrained=google/gemma-3-1b-it,dtype=auto \
    --tasks $tasks \
    --batch_size auto \
    --cache_requests refresh \
    --num_fewshot 0 \
    --output_path ./allres \
    --log_samples > ./allres/gemma-3-1b.log 2>&1 &

nohup CUDA_VISIBLE_DEVICES=4 lm_eval --model vllm \
    --model_args pretrained=google/gemma-3-4b-it,dtype=auto \
    --tasks $tasks \
    --batch_size auto \
    --cache_requests refresh \
    --num_fewshot 0 \
    --output_path ./allres \
    --log_samples > ./allres/gemma-3-4b.log 2>&1 &




