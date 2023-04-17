model_path=""
output_path=""
log_path="" 
model_file=""

CUDA_VISIBLE_DEVICES=0 python -W ignore ./main_eval.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file}