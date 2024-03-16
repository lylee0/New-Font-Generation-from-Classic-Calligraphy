# New-Font-Generation-from-Classic-Calligraphy
Generate a new computer font based on Lanting Xu by Wang Xizhi <br>
Required packages: <br>
Immage size: 80
Total number of characters: 3000
Total training steps: 500,000<br>
LoRA finetuning steps: 100<br>
Content images (for content encoder): content.zip<br>
Style images (for diffusion model and style encoder): <br>
Lantingxu dataset (for LoRA fintune): lantingxu_resized<br>
Use lan.png or wing.png for sampling (or any images of lantingxu)
For adding content encoder:<br>
Modified train.py, sample.py, unet.py, image_datasets.py, train_util.py <br>
For LoRA finetuning (use files in ./lora instead):<br>
Modified lora_train.py, lora.py, fp16_util.py, train_util.py, sample.py
Run lora_train.py and train for 100 steps<br>
