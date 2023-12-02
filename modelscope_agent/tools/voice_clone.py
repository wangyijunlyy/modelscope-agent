import os
import time
from modelscope.models.audio.tts import SambertHifigan
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

import json
import pandas as pd
import requests
from modelscope_agent.tools.tool import Tool, ToolSchema
from pydantic import ValidationError
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3


class VoiceClone(Tool):
    description = '通过用户选定的角色，用特定角色的声音朗读用户输入的文本'
    name = 'voice_clone_generation'
    parameters: list = [{
        'name': 'role',
        'description': '用户想要的角色',
        'required': True
    }, {
        'name': 'text',
        'description':'用户想要输入的想要转换成声音的文本',
        'required': True
    }]

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})
        try:
            all_param = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_param)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')

        self._str = self.tool_schema.model_dump_json()
        self._function = self.parse_pydantic_model_to_openai_function(
            all_param)

    def __call__(self, *args, **kwargs):
        role_list = []
        if kwargs['role'] in role_list:
            model_dir = os.path.abspath(f"./voice_clone_files/{kwargs['role']}_pretrain_work_dir")
            custom_infer_abs = {
                'voice_name':
                'F7',
                'am_ckpt':
                os.path.join(model_dir, 'tmp_am', 'ckpt'),
                'am_config':
                os.path.join(model_dir, 'tmp_am', 'config.yaml'),
                'voc_ckpt':
                os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan', 'ckpt'),
                'voc_config':
                os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan',
                        'config.yaml'),
                'audio_config':
                os.path.join(model_dir, 'data', 'audio_config.yaml'),
                'se_file':
                os.path.join(model_dir, 'data', 'se', 'se.npy')
            }
            voice_kwargs = {'custom_ckpt': custom_infer_abs}

            model_id = SambertHifigan(os.path.join(model_dir, "orig_model"), **voice_kwargs)

            inference = pipeline(task=Tasks.text_to_speech, model=model_id)
            output = inference(input=kwargs['text'])
            return output['output_wav']
            import IPython.display as ipd
            ipd.Audio(output["output_wav"], rate=16000)
        else:
            raise Exception("好像没有这个角色呢，换个角色试试看！")

    

   