import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from utils import dist_util, logger
from utils.script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    create_model_and_diffusion,
)
from PIL import Image
from attrdict import AttrDict
import yaml
from lora import inject_trainable_lora_extended
import itertools


def img_pre_pros(img_path, image_size):
    pil_image = Image.open(img_path).resize((image_size, image_size))
    pil_image.load()
    pil_image = pil_image.convert("RGB")
    arr = np.array(pil_image)
    arr = arr.astype(np.float32) / 127.5 - 1
    return np.transpose(arr, [2, 0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./cfg/test_cfg.yaml',
                        help='config file path')
    parser = parser.parse_args()

    # set up cfg
    with open(parser.cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = AttrDict(create_cfg(cfg))

    model_path = cfg.model_path
    sty_img_path = cfg.sty_img_path
    con_folder_path = cfg.con_folder_path
    gen_text_stroke_path = cfg.gen_text_stroke_path
    total_txt_file = cfg.total_txt_file
    img_save_path = cfg.img_save_path
    classifier_free = cfg.classifier_free
    cont_gudiance_scale = cfg.cont_scale
    sk_gudiance_scale = cfg.sk_scale

    cfg.__delattr__('model_path')
    cfg.__delattr__('sty_img_path')
    cfg.__delattr__('con_folder_path')
    cfg.__delattr__('gen_text_stroke_path')
    cfg.__delattr__('total_txt_file')
    cfg.__delattr__('img_save_path')
    cfg.__delattr__('classifier_free')
    cfg.__delattr__('cont_scale')
    cfg.__delattr__('sk_scale')

    # set up distributed training
    dist_util.setup_dist()

    # save directory
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    # create UNet model and diffusion
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cfg, model_and_diffusion_defaults().keys())
    )

    model.requires_grad_(True)
    unet_lora_params, train_names = inject_trainable_lora_extended(model)

    # load model
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if cfg.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log("sampling...")
    noise = None
    # get words to be generated
    gen_txt=[]
    for file in os.listdir(con_folder_path):
        file_name, file_extension = os.path.splitext(file)
        if file_extension == ".png":
            gen_txt.append(file_name[-1])

    # set up dictionary for trained words
    char2idx = {}
    char_not_in_list = []
    with open(total_txt_file, 'r') as f:
        chars = f.readlines()
        for char in gen_txt:
            if char not in chars[0]:
                chars[0] += char
                char_not_in_list.append(char)
        for idx, char in enumerate(chars[0]):
            char2idx[char] = idx
        f.close()

    # get index of words to be generated
    char_idx = []
    for char in gen_txt:
        char_idx.append(char2idx[char])
    
    all_images = []
    all_labels = []

    stroke_dict = {}
    if gen_text_stroke_path is not None:
        with open(gen_text_stroke_path, 'r') as f:
            gen_text_stroke = f.readlines()
            for gen_strokes in gen_text_stroke:
                stroke_dict[gen_strokes[0]] = gen_strokes[1:]
        f.close()

    # for each batch
    # while len(all_images) * cfg.batch_size < cfg.num_samples:
    ch_idx = 0
    for char in gen_txt:

        model_kwargs = {}

        # process input content image
        con_img = th.tensor(img_pre_pros(con_folder_path + "/" + char + ".png", cfg.image_size), requires_grad=False).cuda().repeat(cfg.batch_size, 1, 1, 1)
        con_feat = model.con_encoder(con_img)
        model_kwargs["y"] = con_feat

        # process target style image
        sty_img = th.tensor(img_pre_pros(sty_img_path, cfg.image_size), requires_grad=False).cuda().repeat(cfg.batch_size, 1, 1, 1)
        sty_feat = model.sty_encoder(sty_img)
        model_kwargs["sty"] = sty_feat

        # process input characters
        classes = th.tensor([i for i in char_idx[ch_idx:ch_idx + cfg.batch_size]], device=dist_util.dev())
        ch_idx += cfg.batch_size

        # read stroke information
        if cfg.stroke_path is not None:
            chars_stroke = th.empty([0, 32], dtype=th.float32)

            # read all stroke
            with open(cfg.stroke_path, 'r') as f:
                lines = f.readlines()
                # need to be in order????
                if gen_text_stroke_path == None:
                    for char in char_not_in_list:
                        lines.append(char + " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
                else:
                    for chn in char_not_in_list:
                        lines.append(chn + stroke_dict[chn])
                for line in lines:
                    strokes = line.split(" ")[1:-1]
                    char_stroke = []
                    for stroke in strokes:
                        char_stroke.append(int(stroke))
                    while len(char_stroke) < 32:  # for korean
                        char_stroke.append(0)
                    assert len(char_stroke) == 32
                    chars_stroke = th.cat((chars_stroke, th.tensor(char_stroke).reshape([1, 32])), dim=0)
            f.close()

            # take needed info
            '''if classes >= 3000:
                stk = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                stkth = th.tensor(stk)
                model_kwargs["stroke"] = stkth.to(dist_util.dev())
            else:'''
            device = dist_util.dev()
            chars_stroke = chars_stroke.to(device)
            model_kwargs["stroke"] = chars_stroke[classes].to(device)
            #model_kwargs["stroke"] = chars_stroke[classes].to(dist_util.dev())

        if classifier_free:
            if cfg.stroke_path is not None:
                model_kwargs["mask_y"] = th.cat([th.zeros([cfg.batch_size], dtype=th.bool), th.ones([cfg.batch_size * 2], dtype=th.bool)]).to(dist_util.dev())
                model_kwargs["y"] = model_kwargs["y"].repeat(3)
                model_kwargs["mask_stroke"] = th.cat(
                    [th.ones([cfg.batch_size], dtype=th.bool),th.zeros([cfg.batch_size], dtype=th.bool), th.ones([cfg.batch_size], dtype=th.bool)]).to(
                    dist_util.dev())
                model_kwargs["stroke"] = model_kwargs["stroke"].repeat(3, 1)
                model_kwargs["sty"] = model_kwargs["sty"].repeat(3, 1)
            else:
                model_kwargs["mask_y"] = th.cat([th.zeros([cfg.batch_size], dtype=th.bool), th.ones([cfg.batch_size], dtype=th.bool)]).to(dist_util.dev())
                model_kwargs["y"] = model_kwargs["y"].repeat(2)
                model_kwargs["sty"] = model_kwargs["sty"].repeat(2, 1)
        else:
            model_kwargs["mask_y"] = th.zeros([cfg.batch_size], dtype=th.bool).to(dist_util.dev())
            if cfg.stroke_path is not None:
                model_kwargs["mask_stroke"] = th.zeros([cfg.batch_size], dtype=th.bool).to(dist_util.dev())

        def model_fn(x_t, ts, **model_kwargs):
            if classifier_free:
                repeat_time = model_kwargs["y"].shape[0] // x_t.shape[0]
                x_t = x_t.repeat(repeat_time, 1, 1, 1)
                ts = ts.repeat(repeat_time)

                if cfg.stroke_path is not None:
                    model_output = model(x_t, ts, **model_kwargs)
                    model_output_y, model_output_stroke, model_output_uncond = model_output.chunk(3)
                    model_output = model_output_uncond + \
                                   cont_gudiance_scale * (model_output_y - model_output_uncond) + \
                                   sk_gudiance_scale * (model_output_stroke - model_output_uncond)

                else:

                    model_output = model(x_t, ts, **model_kwargs)
                    model_output_cond, model_output_uncond = model_output.chunk(2)
                    model_output = model_output_uncond + cont_gudiance_scale * (model_output_cond - model_output_uncond)

            else:
                model_output = model(x_t, ts, **model_kwargs)
            return model_output

        sample_fn = (
            diffusion.p_sample_loop if not cfg.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (cfg.batch_size, 3, cfg.image_size, cfg.image_size),
            clip_denoised=cfg.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            noise=noise,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_labels = [
            th.zeros_like(classes) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * cfg.batch_size} samples")

    # save images
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: cfg.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: cfg.num_samples]
    #char2idx.keys()[char2idx.values().index(label)]
    if dist.get_rank() == 0:
        for idx, (img_sample, img_cls) in enumerate(zip(arr, label_arr)):
            img = Image.fromarray(img_sample).convert("RGB")
            img_name = gen_txt[idx] + ".png"
            #img_name = "%05d.png" % (idx)  #change the name
            img.save(os.path.join(img_save_path, img_name))

    dist.barrier()
    logger.log("sampling complete")


def create_cfg(cfg):
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=16,
        use_ddim=False,
        model_path="",
        cont_scale=1.0,
        sk_scale=1.0,
        sty_img_path="",
        #gen_text_stroke_path="", # add
        stroke_path=None,
        attention_resolutions='40, 20, 10',
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cfg)
    return defaults


if __name__ == "__main__":
    main()
