# New-Font-Generation-from-Classic-Calligraphy
Generate a new computer font based on Lanting Xu by Wang Xizhi <br>
Total training steps: 500,000<br>
LoRA finetuning steps: 100<br>
Content images: <br>
Style images: <br>
Lantingxu dataset: <br>
Use lan.png or wing.png for sampling
For adding content encoder:<br>
Modified train.py, sample.py, unet.py, image_datasets.py, train_util.py <br>
For LoRA finetuning:<br>
Modified lora_train.py, lora.py, fp16_util.py, train_util.py, sample.py
Run lora_train.py and train for 100 steps<br>
