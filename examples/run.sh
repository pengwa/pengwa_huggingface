

recipe="language-modeling/run_mlm.py --model_name_or_path bert-large-uncased --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1"

model_name="bert"

steps=200
mixed_precision="--fp16"
ort="--ort"

batch_size=8

/home/pengwa/nsight-systems-2021.3.1/bin/nsys profile -o huggingface_${model_name}_%p -t cuda,cudnn,cublas,osrt,nvtx python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 examples/pytorch/${recipe} --seed 666 --do_train --output_dir /tmp/output --overwrite_output_dir --max_steps ${steps} --logging_steps ${steps} --logging_first_step --per_device_train_batch_size ${batch_size} --fp16_backend "apex" --fp16_opt_level "O2"  ${mixed_precision} ${deepspeed} ${ort}