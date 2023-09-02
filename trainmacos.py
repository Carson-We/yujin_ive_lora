import subprocess

# Train data path
pretrained_model = "/Users/tszsanwu/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors"
train_data_dir = "/Users/tszsanwu/Desktop/Loras/Image"
reg_data_dir = "/Users/tszsanwu/Desktop/Loras/reg/1_woman"

# Train related params
resolution = "768,768"
batch_size = 2
max_train_epochs = 10
save_every_n_epochs = 1
network_dim = 64
network_alpha = 32
clip_skip = 2
train_unet_only = 0
train_text_encoder_only = 0

# Learning rate
lr = "5e-5"
unet_lr = "5e-5"
text_encoder_lr = "6e-6"
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 50
lr_restart_cycles = 1

# Output settings
output_name = "yujinive_v2"
save_model_as = "safetensors"

network_weights = ""
min_bucket_reso = 256
max_bucket_reso = 1024
persistent_data_loader_workers = 0

# Activate python venv
subprocess.run(["source", "venv/bin/activate"], shell=True)

# Set environment variable
subprocess.run(["export", "HF_HOME=huggingface"], shell=True)

# Run train
subprocess.run([
    "python", "-m", "accelerate", "launch", "--num_processes=1", "--num_workers=8", "--use_env",
    "./sd-scripts/train_network.py",
    "--enable_bucket",
    f"--pretrained_model_name_or_path={pretrained_model}",
    f"--train_data_dir={train_data_dir}",
    "--output_dir=./output",
    "--logging_dir=./logs",
    f"--resolution={resolution}",
    "--network_module=networks.lora",
    f"--max_train_epochs={max_train_epochs}",
    f"--learning_rate={lr}",
    f"--unet_lr={unet_lr}",
    f"--text_encoder_lr={text_encoder_lr}",
    f"--lr_scheduler={lr_scheduler}",
    f"--lr_warmup_steps={lr_warmup_steps}",
    f"--lr_scheduler_num_cycles={lr_restart_cycles}",
    f"--network_dim={network_dim}",
    f"--network_alpha={network_alpha}",
    f"--output_name={output_name}",
    f"--train_batch_size={batch_size}",
    f"--save_every_n_epochs={save_every_n_epochs}",
    "--mixed_precision=fp16",
    "--save_precision=fp16",
    "--seed=1337",
    "--cache_latents",
    f"--clip_skip={clip_skip}",
    "--prior_loss_weight=1",
    "--max_token_length=225",
    "--caption_extension=.txt",
    f"--save_model_as={save_model_as}",
    f"--min_bucket_reso={min_bucket_reso}",
    f"--max_bucket_reso={max_bucket_reso}",
    "--xformers",
    "--shuffle_caption"
])

print("Train finished")
input("Press any key to continue...")
