import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from modelscope_agent.tools.base import BaseTool, register_tool
from scipy.io import wavfile

from .facechain_utils.facechain.inference_talkinghead import (
    SadTalker, text_to_speech_edge)

# from modelscope_agent.tools.tool import Tool, ToolSchema


@register_tool('video_gen')
class SadTalkerTool(BaseTool):
    description = '此工具可以生成人物的讲话视频'
    name = 'video_gen'
    parameters: list = [{
        'name': 'audio_path',
        'type': 'string',
        'description': 'necessary audio path',
        'required': True,
    }, {
        'name': 'speaker_path',
        'type': 'string',
        'description': 'necessary speaker image path',
        'required': True,
    }]

    def launch_pipeline_talkinghead(self,
                                    data,
                                    sid,
                                    preprocess='full',
                                    still_mode=True,
                                    use_enhancer=False,
                                    batch_size=1,
                                    size=256,
                                    pose_style=0,
                                    exp_scale=1.0):
        source_image = sid
        #source_image = gradio.processing_utils.encode_pil_to_bytes(image)
        #audio_data = data
        # 将二进制音频数据转换为numpy数组
        audio_np = np.frombuffer(data, dtype=np.int16)
        output_file_path = '/home/wsco/wyj/Bert-VITS2/output.wav'
        # 指定采样率（以样本/秒为单位）
        sampling_rate = 44100  # 例如：44.1 kHz
        #使用 scipy 的 wavfile.write 函数将音频数据写入 WAV 文件
        wavfile.write(output_file_path, sampling_rate, audio_np)
        #driven_audio = audio_np #numpy
        driven_audio = output_file_path
        gen_video = SadTalker(uuid=None)
        before_queue_size = 0
        inference_done_count = 0
        before_done_count = inference_done_count
        with ProcessPoolExecutor(max_workers=5) as executor:
            future = executor.submit(gen_video, source_image, driven_audio,
                                     preprocess, still_mode, use_enhancer,
                                     batch_size, size, pose_style, exp_scale)
        output = future.result()
        print('视频生成完毕')
        return output

    def call(self, params: str, **kwargs):
        #转为json格式params
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        audio_path = params['audio_path']
        speaker_path = params['speaker_path']
        try:
            with open(audio_path, 'rb') as file:
                # 读取二进制数据
                binary_data = file.read()
            print(type(binary_data))
            video_output_path = self.launch_pipeline_talkinghead(
                data=binary_data, sid=speaker_path)
            video_file_path = Path(video_output_path)
            return video_file_path
        except Exception as e:
            return {'message': str(e)}
