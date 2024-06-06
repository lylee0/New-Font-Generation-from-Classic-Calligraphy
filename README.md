# New-Font-Generation-from-Classic-Calligraphy
Generate a new computer font based on Lanting Xu by Wang Xizhi <br>
Reference: https://github.com/Hxyz-123/Font-diff <br>
Required packages: <br>
Image size: 80<br>
Total number of characters: 3000<br>
Total training steps: 500,000<br>
LoRA finetuning steps: 100<br>
Content images (for content encoder): content.zip<br>
Style images (for diffusion model and style encoder): https://mycuhk-my.sharepoint.com/:u:/g/personal/1155158772_link_cuhk_edu_hk/EQXTMdCmKWdNmF4dcI_QleAB_ZUo39Ib0wHSovFu_CePLg?e=wUQefi<br>
Lantingxu dataset (for LoRA fintune): lantingxu_resized<br>
Use lan.png or wing.png for sampling (or any images of lantingxu)<br>
<br>
For adding content encoder:<br>
Modified train.py, sample.py, unet.py, image_datasets.py, train_util.py <br>
<br>
For LoRA finetuning (use files in ./lora instead):<br>
Modified lora_train.py, lora.py, fp16_util.py, train_util.py, sample.py<br>
Run lora_train.py and train for 100 steps<br>
<br>
Finetune without using LoRA:<br>
Train the model for extra 8,000 steps with the lantingxu dataset only<br>
<br>
Image Upscaling:<br>
Run upscale.py in upscale<br>
Put the sample images in ./images<br>
<br>
Remove image background<br>
Run remove_backgound.py<br>

Lanting Xu with template<br>
![Alt text](./images/with_template.png?raw=true)

<br>Black and white Lanting Xu<br>
![Alt text](./images/black.png?raw=true)

<br>Red and yellow Lanting Xu<br>
![Alt text](./images/red_yellow.png?raw=true)

<br>Red and black Lanting Xu<br>
![Alt text](./images/red_black.png?raw=true)

<br>Red and white Lanting Xu<br>
![Alt text](./images/white_red.png?raw=true)
