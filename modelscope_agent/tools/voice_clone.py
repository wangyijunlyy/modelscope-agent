import os
import shutil

import numpy as np
from modelscope_agent.tools.base import BaseTool, register_tool
from scipy.io import wavfile

from modelscope.metainfo import Trainers
from modelscope.models.audio.tts import SambertHifigan
from modelscope.pipelines import pipeline
from modelscope.tools import run_auto_label
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import TtsTrainType
from modelscope.utils.constant import Tasks

WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


def auto_label(wav_path, user_id):
    # make a new dir to save wav sudio
    wav_directory = os.path.join(WORK_DIR, f'wav_dir_{user_id}')
    os.makedirs(wav_directory, exist_ok=True)
    # move wav file to a new dir
    target_path = os.path.join(wav_directory, os.path.basename(wav_path))
    shutil.copy(wav_path, target_path)
    output_data_dir = os.path.join(WORK_DIR, f'output_data_{user_id}')
    os.makedirs(output_data_dir, exist_ok=True)
    ret, report = run_auto_label(
        input_wav=wav_directory,
        work_dir=output_data_dir,
        resource_revision='v1.0.7')
    return output_data_dir


def train(dataset_path, user_id):
    pretrained_model_id = 'damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k'
    dataset_id = dataset_path
    pretrain_work_dir = os.path.join(WORK_DIR, f'pretrain_{user_id}')
    os.makedirs(pretrain_work_dir, exist_ok=True)
    # 训练信息，用于指定需要训练哪个或哪些模型，这里展示AM和Vocoder模型皆进行训练
    # 目前支持训练：TtsTrainType.TRAIN_TYPE_SAMBERT, TtsTrainType.TRAIN_TYPE_VOC
    # 训练SAMBERT会以模型最新step作为基础进行finetune
    train_info = {
        TtsTrainType.TRAIN_TYPE_SAMBERT: {  # 配置训练AM（sambert）模型
            'train_steps': 202,  # 训练多少个step
            'save_interval_steps': 200,  # 每训练多少个step保存一次checkpoint
            'log_interval': 10  # 每训练多少个step打印一次训练日志
        }
    }

    # 配置训练参数，指定数据集，临时工作目录和train_info
    kwargs = dict(
        model=pretrained_model_id,  # 指定要finetune的模型
        model_revision='v1.0.6',
        work_dir=pretrain_work_dir,  # 指定临时工作目录
        train_dataset=dataset_id,  # 指定数据集id
        train_type=train_info  # 指定要训练类型及参数
    )

    trainer = build_trainer(
        Trainers.speech_kantts_trainer, default_args=kwargs)

    trainer.train()
    return pretrain_work_dir


def infer(model_dir, text, user_id):
    # model_dir = os.path.abspath(pretrain_work_dir)
    custom_infer_abs = {
        'voice_name':
        'F7',
        'am_ckpt':
        os.path.join(model_dir, 'tmp_am', 'ckpt'),
        'am_config':
        os.path.join(model_dir, 'tmp_am', 'config.yaml'),
        'voc_ckpt':
        os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan',
                     'ckpt'),
        'voc_config':
        os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan',
                     'config.yaml'),
        'audio_config':
        os.path.join(model_dir, 'data', 'audio_config.yaml'),
        'se_file':
        os.path.join(model_dir, 'data', 'se', 'se.npy')
    }
    kwargs = {'custom_ckpt': custom_infer_abs}

    model_id = SambertHifigan(os.path.join(model_dir, 'orig_model'), **kwargs)

    inference = pipeline(task=Tasks.text_to_speech, model=model_id)
    output = inference(input=text)
    output_wav_path = f'output_audio_{user_id}.wav'
    output_wav_bytes = output['output_wav']
    output_wav_np = np.frombuffer(output_wav_bytes, dtype=np.int16)
    wavfile.write(output_wav_path, 16000, output_wav_np)
    return output_wav_path


@register_tool('voice_gen')
class VoiceCloneTool(BaseTool):
    description = '根据用户输入的参考音频，克隆生成对应文本的音频。'
    name = 'voice_gen'
    parameters: list = [{
        'name': 'audio_path',
        'type': 'string',
        'description': '输入的参考音频文件路径',
        'required': True,
    }, {
        'name': 'text',
        'type': 'string',
        'description': '用户想要生成音频的文本内容',
        'required': True,
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        if params['audio_path'] is None:
            raise ValueError('音频文件上传出错')
        if params['text'] is None:
            raise ValueError('没有获取到音频文本内容')
        user_id = params['audio_path'].split('_')[-1].split('.')[0]
        params['audio_path'] = os.path.join(WORK_DIR, params['audio_path'])
        print('---------------------\nstart auto label')
        dataset_path = auto_label(params['audio_path'], user_id)
        print('---------------------\nstart train')
        pretrain_work_dir = train(dataset_path, user_id)
        print('---------------------\nstart infer（audio gen)')
        output_wav_path = infer(pretrain_work_dir, params['text'], user_id)
        return output_wav_path
