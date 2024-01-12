import os
import platform
import re
import shutil
import subprocess
from difflib import SequenceMatcher
from typing import Dict, Optional

import cv2
import json
import numpy as np
import slugify
import torch

from modelscope import snapshot_download
from .base import BaseTool, register_tool
from .facechain_utils.facechain.inference import GenPortrait, data_process_fn
from .facechain_utils.facechain.utils import snapshot_download

WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')
abs_path = os.path.dirname(os.path.abspath(__file__))
userid_txt_path = './modelscope_agent/tools/facechain_utils/userid.txt'


@register_tool('photo_infer')
class Photo_infer(BaseTool):
    description = '根据用户提供的图片训练出Lora,并且根据训练得到的用户人脸lora和风格lora生成写真'
    name = 'photo_infer'
    parameters: list = [
        {
            'name': 'style_name',
            'description': '用户所需风格的名字',
            'required': True,
            'type': 'string'
        },
        {
            'name': 'style_file',
            'description': '用户传入的风格文件',
            'required': True,
            'type': 'string'
        },
    ]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            params = eval(params)
        if params == None:
            raise ValueError('检查传入参数')
        else:

            if 'style_file' in params:
                # 用户传入的是一个文件
                style_file = params['style_file']
                style_path = style_file
            if 'style_name' in params:
                # 用户传入的是一个风格名字
                style_name = params['style_name']
                # 风格匹配，找到对应的JSON文件

                base_style_path = os.path.join(
                    abs_path, 'facechain_utils/styles/MajicmixRealistic_v6')
                best_similarity = 0  # 用于存储最佳匹配的相似性度量值

                files = os.listdir(base_style_path)
                files.sort()
                for file in files:
                    file_path = os.path.join(base_style_path, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        match_style_name = data.get('name', '')  # 获取风格文件中的名称字段
                        style_chinese_name = re.search(
                            r'[\u4e00-\u9fa5]+', match_style_name).group()
                        print('wwwwwwwwww', style_chinese_name)
                        # 计算文本与风格名称之间的相似性
                        similarity = SequenceMatcher(
                            None, style_name, style_chinese_name).ratio()

                        # 如果相似性高于当前最佳匹配，更新最佳匹配
                        if similarity > best_similarity:
                            best_similarity = similarity
                            style_path = file_path  # 更新最佳匹配的文件路径

        # 根据目录的useridtxt文档的提取出userid
        # instance_data_dir的问题需要解决
        with open(userid_txt_path, 'r') as f:
            userid = f.read()
        # 根据userid找到LoRA文件，不需要用户传入了
        user_LoRA_path = os.path.join(
            './modelscope_agent/tools/facechain_utils/qw/ly261666/cv_portrait_model',
            userid)
        result = self._interface(
            matched_style_file_path=style_path,
            user_model=userid,
            user_LoRA_path=user_LoRA_path)

        # 待完成======================================================================================
        return result

    def _interface(self, matched_style_file_path: str, user_model: str,
                   user_LoRA_path: str):
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
                Lora_model_path=user_LoRA_path)
        except Exception as e:
            import traceback
            print(f'error {e} with detail {traceback.format_exc()}')

        return os.path.join(single_path, '0.png')


@register_tool('photo_gen')
class FacechainTool(BaseTool):
    description = '根据用户提供的图片训练出Lora,并且根据训练得到的用户人脸lora和风格lora生成写真'
    name = 'photo_gen'
    parameters: list = [{
        'name': 'image_path',
        'description': '用户传入的图片路径',
        'required': True,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        if params == None:
            raise ValueError('请传入图片路径')
        else:
            path = params['image_path']
            userid = f"w{params['image_path'].split('_')[-1].split('.')[0]}"
        path = os.path.join(WORK_DIR, path)

        # 以覆盖的方式创建一个txt文档用于存储userid，在photo_infer工具中用于找出人物LoRA的路径以及人物图片的位置
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(os.path.dirname(userid_txt_path), exist_ok=True)
        # 创建并打开文件，如果文件已存在则覆盖原文件内容
        with open(userid_txt_path, 'w') as file:
            file.write(str(userid))  # 在文件中写入id

        finetunetool = FaceChainFineTuneTool(lora_name=userid)
        finetunetool.call(path)
        interface_tool = FaceChainInferenceTool(user_model=userid)
        result = interface_tool.call()
        return result


class FaceChainFineTuneTool():

    def __init__(self, lora_name: str):
        super().__init__()
        self.base_model_path = 'ly261666/cv_portrait_model'
        self.revision = 'v2.0'
        self.sub_path = 'film/film'
        # 这里固定了Lora的名字,重新训练会覆盖原来的
        self.lora_name = lora_name

    def call(self, path: str) -> str:
        uuid = 'qw'
        _train_lora(uuid, self.lora_name, self.base_model_path, self.revision,
                    self.sub_path, path)


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

    lora_path = os.path.join(
        abs_path, 'facechain_utils/facechain/train_text_to_image_lora.py')

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
    Lora_model_path=None,
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

    print('-------user_model(也就是人物lora name): ', user_model)

    use_main_model = True
    use_face_swap = True
    use_post_process = True
    use_stylization = False
    # user_model就是人物lora的name
    instance_data_dir = os.path.join(
        './modelscope_agent/tools/facechain_utils/',
        character_model.split('/')[-1], user_model)
    if Lora_model_path == None:
        lora_model_path = f'./modelscope_agent/tools/facechain_utils/{uuid}/{character_model}/{user_model}'
    else:
        lora_model_path = Lora_model_path
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
    if lora_choice == 'preset':
        save_dir = os.path.join(save_dir, 'style_' + style_model[:2])
    else:
        save_dir = os.path.join(
            save_dir, 'lora_' + os.path.basename(lora_choice).split('.')[0])
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

    if len(outputs) > 0:
        result = concatenate_images(outputs)
        shutil.rmtree(os.path.join(save_dir, 'concat'), ignore_errors=True)
        if not os.path.exists(os.path.join(save_dir, 'concat')):
            os.makedirs(os.path.join(save_dir, 'concat'))
        num = len(os.listdir(os.path.join(save_dir, 'concat')))
        image_path = os.path.join(save_dir, 'concat', str(num) + '.png')
        cv2.imwrite(image_path, result)  #整体图像

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


class FaceChainInferenceTool():
    # user_model 也就是 lora_name
    def __init__(self, user_model: str):
        self.user_model = user_model
        super().__init__()

    def _inference(self, matched_style_file_path: str):
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
                user_model=self.user_model,
                num_images=1,
                multiplier_style=0.35,
                multiplier_human=0.95,
                pose_model=None,
                pose_image=None,
                lora_choice='preset')
        except Exception as e:
            import traceback
            print(f'error {e} with detail {traceback.format_exc()}')

        return os.path.join(single_path, '0.png')

    def call(self) -> str:
        style_path = os.path.join(
            abs_path,
            'facechain_utils/styles/MajicmixRealistic_v6/Chinese_New_Year.json'
        )
        result = self._inference(matched_style_file_path=style_path)
        return result
