#!/bin/bash

# --- 配置 ---
# 1. 定义任务列表 (使用 Bash 数组)
    # bbh
    # truthfulqa
    # agieval
    # mmlu
    # ifeval
    # arc_easy
    # arc_challenge
    # hellaswag
    # openbookqa
    # piqa
    # social_iqa
    # winogrande
    # wikitext
    # gpqa
tasks=(
    minerva_math
    gsm8k
    humaneval  # Code generation often benefits from batch_size=1
    mbpp       # Code generation often benefits from batch_size=1
)

# 2. 定义模型和其他固定参数
MODEL_NAME="google/gemma-3-1b-it"
NUM_FEWSHOT=0
BASE_OUTPUT_DIR="/mnt/yizhenjia3/routellm-distl/allres_detailed/gemma3-1b" # 基础输出目录
DEVICE_ID=4

# 3. 定义任务特定的 Batch Size (使用 Associative Array)
#    Key: task name, Value: desired batch size
declare -A task_batch_sizes=(
    [bbh]=32
    [agieval]=16
    [ifeval]=128
    [arc_easy]=128
    [arc_challenge]=128
    [hellaswag]=128
    [openbookqa]=128
    [piqa]=128
    [social_iqa]=128
    [winogrande]=128
    [wikitext]=128
    [gpqa]=1
    [minerva_math]=32
    [gsm8k]=32
    [humaneval]=32
    [mbpp]=32
)

# 4. 定义默认 Batch Size (用于未在上面明确指定的任务)
DEFAULT_BATCH_SIZE="auto"

# --- 准备工作 ---
# 确保基础输出目录存在
mkdir -p "$BASE_OUTPUT_DIR"

# 设置环境变量
export HF_ALLOW_CODE_EVAL="1"
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

# --- 主循环 ---
echo "Starting evaluation loop for ${#tasks[@]} tasks..."

for task in "${tasks[@]}"; do
    echo "--------------------------------------------------"
    echo "$(date): Starting task: $task"
    echo "--------------------------------------------------"

    # 确定当前任务的 Batch Size
    # 检查 task_batch_sizes 中是否有该任务的条目
    # 如果有，则使用指定的值；如果没有，则使用 DEFAULT_BATCH_SIZE
    current_batch_size=${task_batch_sizes[$task]:-$DEFAULT_BATCH_SIZE}
    echo "Using Batch Size: $current_batch_size (Default: $DEFAULT_BATCH_SIZE)"


    # 为当前任务创建独立的输出和日志目录
    TASK_OUTPUT_DIR="$BASE_OUTPUT_DIR/$task"
    LOG_FILE="$TASK_OUTPUT_DIR/${task}.log"
    mkdir -p "$TASK_OUTPUT_DIR" # 确保任务子目录存在

    # 构建当前任务的 lm_eval 命令 (使用 current_batch_size)
    command="lm_eval --model vllm \
        --model_args pretrained=$MODEL_NAME \
        --tasks $task \
        --batch_size $current_batch_size \
        --cache_requests refresh \
        --num_fewshot $NUM_FEWSHOT \
        --output_path $TASK_OUTPUT_DIR \
        --log_samples \
        --confirm_run_unsafe_code"

    echo "Running command:"
    echo "$command"
    echo "Logging to: $LOG_FILE"

    # 执行命令，并将 stdout 和 stderr 重定向到独立的日志文件
    $command > "$LOG_FILE" 2>&1

    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "$(date): Task $task completed successfully."
    else
        echo "$(date): Task $task FAILED. Check log: $LOG_FILE. Continuing to next task."
    fi
    echo # 添加空行以提高可读性

done

echo "=================================================="
echo "$(date): All tasks processed."
echo "Results and logs saved in subdirectories under $BASE_OUTPUT_DIR"
echo "=================================================="