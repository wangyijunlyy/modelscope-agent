import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from modelscope_agent.tools.base import BaseTool, register_tool
from .facechain_utils.facechain.inference_talkinghead import SadTalker
from modelscope_agent.tools.utils.output_wrapper import VideoWrapper
def transcode_to_h264(input_file_path, output_file_path):
    try:
        # 使用 FFmpeg 将视频重新编码为 H.264
        subprocess.run([
            'ffmpeg',
            '-i', input_file_path,        # 输入文件路径
            '-c:v', 'libx264',            # 指定视频编码器为 libx264 (H.264)
            '-crf', '23',                  # 设置视频质量
            '-c:a', 'aac',                 # 指定音频编码器为 AAC
            '-strict', 'experimental',     # 启用实验性音频编码器，对一些老版本的 FFmpeg 可能需要
            '-b:a', '192k',                # 设置音频比特率
            output_file_path,             # 输出文件路径
            '-y'
        ])

        print("转码成功")
    except Exception as e:
        print(f"转码失败: {e}")

@register_tool('video_gen')
class SadTalkerTool(BaseTool):
    description = '此工具可以生成人物的讲话视频'
    name = 'video_gen'
    parameters: list = [{
        'name': 'user_id',
        'type': 'string',
        'description': '用户id',
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
        driven_audio = data
        gen_video = SadTalker(uuid=None)
        # with ProcessPoolExecutor(max_workers=5) as executor:
        #     future = executor.submit(gen_video, source_image, driven_audio,
        #                              preprocess, still_mode, use_enhancer,
        #                              batch_size, size, pose_style, exp_scale)
        output = gen_video(source_image, driven_audio,
                                     preprocess, still_mode, use_enhancer,
                                     batch_size, size, pose_style, exp_scale)
        print('视频生成完毕')
        return output

    def call(self, params: str, **kwargs):
        #转为json格式params
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        if params['user_id'] == None:
            raise ValueError('没有获取到用户id')
        audio_path = f"wav_utils/output_audio_{params['user_id']}.wav"
        speaker_image_path = f"modelscope_agent/tools/facechain_utils/{params['user_id']}/single/0.png"
        if os.path.exists(audio_path):
            print(f"Audio path exists: {audio_path}")
        else:
            raise ValueError('audio_path错误')

        if os.path.exists(speaker_image_path):
            print(f"Speaker image path exists: {speaker_image_path}")
        else:
           raise ValueError('speaker_image_path错误')       
        video_output_path = self.launch_pipeline_talkinghead(
            data=audio_path, sid=speaker_image_path)
        video_output_path_out = os.path.join(os.path.dirname(video_output_path),f"out_{params['user_id']}.wav")
        transcode_to_h264(video_output_path, video_output_path_out)
        # video_file_path = Path(video_output_path)
        return VideoWrapper(video_output_path_out)
        
