import os
import platform
import shutil
import json
import subprocess
import cv2
import numpy as np
import slugify
import torch
from modelscope import snapshot_download
from .facechain.inference import GenPortrait, data_process_fn
from .facechain.utils import snapshot_download
abs_dir = os.path.dirname(os.path.abspath(__file__))
def train_lora_fn(base_model_path=None,
                  revision=None,
                  sub_path=None,
                  output_img_dir=None,
                  work_dir=None,
                  photo_num=0):
    torch.cuda.empty_cache()

    lora_r = 4
    lora_alpha = 32
    max_train_steps = min(photo_num * 200, 800)

    lora_path = os.path.join(abs_dir, 'facechain/train_text_to_image_lora.py')

    if platform.system() == 'Windows':
        command = [
            'accelerate', 'launch', 'facechain/train_text_to_image_lora.py',
            f'--pretrained_model_name_or_path={base_model_path}',
            f'--revision={revision}', f'--sub_path={sub_path}',
            f'--output_dataset_name={output_img_dir}', '--caption_column=text',
            '--resolution=512', '--random_flip', '--train_batch_size=1',
            '--num_train_epochs=200', '--checkpointing_steps=5000',
            '--learning_rate=1.5e-04', '--lr_scheduler=cosine',
            '--lr_warmup_steps=0', '--seed=42', f'--output_dir={work_dir}',
            f'--lora_r={lora_r}', f'--lora_alpha={lora_alpha}',
            '--lora_text_encoder_r=32', '--lora_text_encoder_alpha=32',
            '--resume_from_checkpoint="fromfacecommon"'
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error executing the command: {e}')
            raise RuntimeError('训练失败 (Training failed)')
    else:
        res = os.system(f'PYTHONPATH=. accelerate launch {lora_path} '
                        f'--pretrained_model_name_or_path={base_model_path} '
                        f'--revision={revision} '
                        f'--sub_path={sub_path} '
                        f'--output_dataset_name={output_img_dir} '
                        f'--caption_column="text" '
                        f'--resolution=512 '
                        f'--random_flip '
                        f'--train_batch_size=1 '
                        f'--num_train_epochs=200 '
                        f'--checkpointing_steps=5000 '
                        f'--learning_rate=1.5e-04 '
                        f'--lr_scheduler="cosine" '
                        f'--lr_warmup_steps=0 '
                        f'--seed=42 '
                        f'--output_dir={work_dir} '
                        f'--lora_r={lora_r} '
                        f'--lora_alpha={lora_alpha} '
                        f'--lora_text_encoder_r=32 '
                        f'--lora_text_encoder_alpha=32 '
                        f'--resume_from_checkpoint="fromfacecommon"')
        if res != 0:
            raise RuntimeError('训练失败 (Training failed)')


def _train_lora(uuid, output_model_name, base_model_path, revision, sub_path,
                path):
    output_model_name = slugify.slugify(output_model_name)
    if not os.path.exists(f'./modelscope_agent/tools/facechain_utils/{uuid}'):
        os.makedirs(f'./modelscope_agent/tools/facechain_utils/{uuid}')
    # mv user upload data to target dir
    instance_data_dir = os.path.join(
        './modelscope_agent/tools/facechain_utils/',
        base_model_path.split('/')[-1], output_model_name)
    print('################################instance_data_dir',
          instance_data_dir)
    work_dir = f'./modelscope_agent/tools/facechain_utils/{uuid}/{base_model_path}/{output_model_name}'
    print('################################work_dir', work_dir)
    labeled_path = instance_data_dir + '_labeled'
    if not os.path.exists(instance_data_dir):
        os.makedirs(instance_data_dir)
    if not os.path.exists(labeled_path):
        os.makedirs(labeled_path)

    shutil.rmtree(work_dir, ignore_errors=True)

    newpath = os.path.join(instance_data_dir, path.split('/')[-1])
    # 将用户的图片路径中的图片移动到对应的文件夹下
    shutil.copy(path, newpath)

    data_process_fn(instance_data_dir, True)

    train_lora_fn(
        base_model_path=base_model_path,
        revision=revision,
        sub_path=sub_path,
        output_img_dir=instance_data_dir,
        work_dir=work_dir,
        photo_num=len(instance_data_dir))


training_done_count = 0
inference_done_count = 0
base_models = [
    {
        'name': 'leosamsMoonfilm_filmGrain20',
        'model_id': 'ly261666/cv_portrait_model',
        'revision': 'v2.0',
        'sub_path': 'film/film'
    },
    {
        'name': 'MajicmixRealistic_v6',
        'model_id': 'YorickHe/majicmixRealistic_v6',
        'revision': 'v1.0.0',
        'sub_path': 'realistic'
    },
]
neg_prompt = '(nsfw:2), paintings, sketches, (worst quality:2), (low quality:2), ' \
             'lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, bad hand, tattoo, (username, watermark, signature, time signature, timestamp, artist name, copyright name, copyright),'\
             'low res, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, extra fingers, fewer fingers, strange fingers, bad hand, mole, ((extra legs)), ((extra hands))'
pos_prompt_with_cloth = 'raw photo, masterpiece, chinese, {}, solo, medium shot, high detail face, looking straight into the camera with shoulders parallel to the frame, photorealistic, best quality'
pos_prompt_with_style = '{}, upper_body, raw photo, masterpiece, solo, medium shot, high detail face, photorealistic, best quality'


def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0],
                           x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image


def launch_pipeline(
    uuid,
    pos_prompt,
    matched,
    num_images,
    neg_prompt=None,
    base_model_index=0,
    user_model=None,
    lora_choice=None,
    multiplier_style=0.35,
    multiplier_human=0.95,
    pose_model=None,
    pose_image=None,
):
    uuid = 'qw'
    character_model = 'ly261666/cv_portrait_model'  #
    # Check character LoRA
    # =======================================================================================================================================
    folder_path = f'./modelscope_agent/tools/facechain_utils/{uuid}/{character_model}'
    # =======================================================================================================================================
    folder_list = []
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isdir(folder_path):
                file_lora_path = f'{file_path}/pytorch_lora_weights.bin'
                if os.path.exists(file_lora_path):
                    folder_list.append(file)
    if len(folder_list) == 0:
        raise '没有人物LoRA，请先训练(There is no character LoRA, please train first)!'
    # Check output model
    if user_model is None:
        raise '请选择人物LoRA(Please select the character LoRA)！'
    base_model = base_models[base_model_index]['model_id']
    revision = base_models[base_model_index]['revision']
    sub_path = base_models[base_model_index]['sub_path']

    style_model = matched['name']
    if matched['model_id'] is None:
        style_model_path = None
    else:
        model_dir = snapshot_download(
            matched['model_id'], revision=matched['revision'])
        style_model_path = os.path.join(model_dir, matched['bin_file'])

    if pose_image is None or pose_model == 0:
        pose_model_path = None
        use_depth_control = False
        pose_image = None
    else:
        model_dir = snapshot_download(
            'damo/face_chain_control_model', revision='v1.0.1')
        pose_model_path = os.path.join(
            model_dir, 'model_controlnet/control_v11p_sd15_openpose')
        if pose_model == 1:
            use_depth_control = True
        else:
            use_depth_control = False

    print('-------user_model(也就是人物id): ', user_model)

    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False
    instance_data_dir = os.path.join(
        './modelscope_agent/tools/facechain_utils/',
        character_model.split('/')[-1], user_model)
    
    lora_model_path = f'./modelscope_agent/tools/facechain_utils/{uuid}/{character_model}/{user_model}'
    
    gen_portrait = GenPortrait(pose_model_path, pose_image, use_depth_control,
                               pos_prompt, neg_prompt, style_model_path,
                               multiplier_style, multiplier_human,
                               use_main_model, use_face_swap, use_post_process,
                               use_stylization)
    num_images = min(6, num_images)
    outputs = gen_portrait(instance_data_dir, num_images, base_model,
                           lora_model_path, sub_path, revision)
    outputs_RGB = []
    for out_tmp in outputs:
        outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))

    save_dir = os.path.join('./modelscope_agent/tools/facechain_utils/',
                            user_model)
    # 不区分风格
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # use single to save outputs
    shutil.rmtree(
        os.path.join(save_dir, 'single'),
        ignore_errors=True)  #删除同一个用户同一个风格上一次生成的照片
    if not os.path.exists(os.path.join(save_dir, 'single')):
        os.makedirs(os.path.join(save_dir, 'single'))
    single_path = os.path.join(save_dir, 'single')
    for img in outputs:
        # count the number of images in the folder
        num = len(os.listdir(os.path.join(save_dir, 'single')))
        cv2.imwrite(os.path.join(save_dir, 'single', str(num) + '.png'), img)

        return ('生成完毕(Generation done)!', outputs_RGB, single_path)
    else:
        return ('生成失败, 请重试(Generation failed, please retry)!', outputs_RGB,
                single_path)


def generate_pos_prompt(matched_style_file, prompt_cloth):
    if matched_style_file is not None:
        if matched_style_file['model_id'] is None:
            pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
        else:
            pos_prompt = pos_prompt_with_style.format(
                matched_style_file['add_prompt_style'])
    else:
        pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
    return pos_prompt

def _interface(matched_style_file_path: str, user_model: str,
                   ):
        with open(matched_style_file_path, 'r') as f:
            matched_style_file = json.load(f)
        try:
            pos_prompt = generate_pos_prompt(
                matched_style_file, matched_style_file['add_prompt_style'])
            if 'leosamsMoonfilm_filmGrain20' in matched_style_file_path:
                base_model_index = 0
            elif 'MajicmixRealistic_v6' in matched_style_file_path:
                base_model_index = 1
            (infer_progress, output_images, single_path) = launch_pipeline(
                uuid='qw',
                matched=matched_style_file,
                pos_prompt=pos_prompt,
                neg_prompt=neg_prompt,
                base_model_index=base_model_index,
                user_model=user_model,
                num_images=1,
                multiplier_style=0.35,
                multiplier_human=0.95,
                pose_model=None,
                pose_image=None,
                lora_choice='preset',
                )
        except Exception as e:
            import traceback
            print(f'error {e} with detail {traceback.format_exc()}')

        return os.path.abspath(os.path.join(single_path, '0.png'))
