# Train data path 
$pretrained_model="/Users/tszsanwu/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors" # base model path 
$train_data_dir="/Users/tszsanwu/Desktop/Loras/Image"
$reg_data_dir = "" # directory for regularization images 

# Train related params 
$resolution = "768,768" # image resolution w,h.
$batch_size = 2 # batch size
$max_train_epoches = 10 # max train epoches 
$save_every_n_epochs = 1 # save every n epochs 
$network_dim = 64 # network dim 
$network_alpha = 32 # network alpha
$clip_skip = 2 # clip skip 
$train_unet_only = 0 # train U-Net only 
$train_text_encoder_only = 0 # train Text Encoder only 

# Learning rate 
$lr = "5e-5"
$unet_lr = "5e-5"
$text_encoder_lr = "6e-6"
$lr_scheduler = "cosine_with_restarts" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
$lr_warmup_steps = 50 # warmup steps 
$lr_restart_cycles = 1 # cosine_with_restarts restart cycles 

# Output settings 
$output_name = "yujinive_v2" # output model name 
$save_model_as = "safetensors" # model save ext 

$network_weights = "" # pretrained weights for LoRA network 
# $network_weights = "D:\workspace\stable-diffusion-webui\models\Lora\koreanDollLikeness_v10.safetensors" # pretrained weights for LoRA network 
$min_bucket_reso = 256 # arb min resolution 
$max_bucket_reso = 1024 # arb max resolution 
$persistent_data_loader_workers = 0 # persistent dataloader workers 

$use_8bit_adam = 0 # use 8bit adam optimizer 
$use_lion = 1 # use lion optimizer 

# Activate python venv
.\venv\Scripts\activate

$Env:HF_HOME = "huggingface"
$ext_args = [System.Collections.ArrayList]::new()

if ($train_unet_only) {
  [void]$ext_args.Add("--network_train_unet_only")
}

if ($train_text_encoder_only) {
  [void]$ext_args.Add("--network_train_text_encoder_only")
}

if ($network_weights) {
  [void]$ext_args.Add("--network_weights=" + $network_weights)
}

if ($reg_data_dir) {
  [void]$ext_args.Add("--reg_data_dir=" + $reg_data_dir)
}

if ($use_8bit_adam) {
  [void]$ext_args.Add("--use_8bit_adam")
}

if ($use_lion) {
  [void]$ext_args.Add("--use_lion_optimizer")
}

if ($persistent_data_loader_workers) {
  [void]$ext_args.Add("--persistent_data_loader_workers")
}

# run train
accelerate launch --num_cpu_threads_per_process=8 "./sd-scripts/train_network.py" `
  --enable_bucket `
  --pretrained_model_name_or_path=$pretrained_model `
  --train_data_dir=$train_data_dir `
  --output_dir="./output" `
  --logging_dir="./logs" `
  --resolution=$resolution `
  --network_module=networks.lora `
  --max_train_epochs=$max_train_epoches `
  --learning_rate=$lr `
  --unet_lr=$unet_lr `
  --text_encoder_lr=$text_encoder_lr `
  --lr_scheduler=$lr_scheduler `
  --lr_warmup_steps=$lr_warmup_steps `
  --lr_scheduler_num_cycles=$lr_restart_cycles `
  --network_dim=$network_dim `
  --network_alpha=$network_alpha `
  --output_name=$output_name `
  --train_batch_size=$batch_size `
  --save_every_n_epochs=$save_every_n_epochs `
  --mixed_precision="fp16" `
  --save_precision="fp16" `
  --seed="1337" `
  --cache_latents `
  --clip_skip=$clip_skip `
  --prior_loss_weight=1 `
  --max_token_length=225 `
  --caption_extension=".txt" `
  --save_model_as=$save_model_as `
  --min_bucket_reso=$min_bucket_reso `
  --max_bucket_reso=$max_bucket_reso `
  --xformers --shuffle_caption $ext_args

Write-Output "Train finished"
Read-Host | Out-Null ;

