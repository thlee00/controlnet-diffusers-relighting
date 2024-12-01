nvidia-smi
label=${0/launch_/}
label=${label/.sh/}
accelerate launch --mixed_precision bf16 train_controlnet_with_albedo.py --output_dir=output/${BASH_SOURCE/launch_/}/$(date -Ins) --validation_image {everett_kitchen{5,9},everett_lobby{1,16}}_dir_{23,14,18,10}.jpg --train_batch_size 8 --images_dir multilum_images/1536x1024 --inject_lighting_direction --concat_albedo_maps --dropout_rgb 0.1 --dir_sh 4 --checkpointing_steps 25000 --max_train_steps 150000 "${@}"



accelerate launch --mixed_precision bf16 train_controlnet_with_albedo.py --controlnet_model_name_or_path /workspace/controlnet-diffusers-relighting/weights/controlnet --output_dir=output/${BASH_SOURCE/launch_/}/$(date -Ins) --validation_image {14n_office{1,3},14n_copyroom{1,6}}_dir_{23,14,18,10}.jpg --train_batch_size 2 --images_dir multilum_images/1536x1024 --inject_lighting_direction --concat_albedo_maps --dropout_rgb 0.1 --dir_sh 4 --checkpointing_steps 5000 --max_train_steps 150000 "${@}" |& tee -a log.txt