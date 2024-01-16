import os
from .base import BaseTool, register_tool
from .facechain_utils.facechain.inference import GenPortrait, data_process_fn
from .facechain_utils.utils import _interface,_train_lora
from modelscope_agent.tools.utils.output_wrapper import ImageWrapper
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')
abs_dir = os.path.dirname(os.path.abspath(__file__))

@register_tool('photo_infer')
class PhotoInfer(BaseTool):
    description = '根据用户选定的风格生成新的写真照'
    name = 'photo_infer'
    parameters: list = [
        {
            'name': 'user_id',
            'description': '用户id',
            'required': True,
            'type': 'string'
        },
        # {
        #     'name': 'style_name',
        #     'description': '用户所需风格的名字',
        #     'required': True,
        #     'type': 'string'
        # },
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
            return 'Parameter Error'
        if params['style_file'] is None and params['style_name'] is None:
            raise ValueError('没有获取到用户选择的风格')
        else:
            if params['style_file']:
                # 用户通过前端点击传入的是一个文件路径
                style_path = params['style_file']

        userid = params['user_id']
        result = _interface(
            matched_style_file_path=style_path,
            user_model=userid,
            )
        return ImageWrapper(result)
    
@register_tool('photo_gen')
class PhotoGen(BaseTool):
    description = '根据用户提供的图片训练出Lora,并且根据训练生成一张默认风格写真照，并且返回user_id。'
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
            userid = params['image_path'].split('_')[-1].split('.')[0]
        path = os.path.join(WORK_DIR, path)
        _train_lora('qw', userid, 'ly261666/cv_portrait_model', 'v2.0',
                    'film/film', path)
        result = _interface(matched_style_file_path=os.path.join(abs_dir,'facechain_utils/styles/MajicmixRealistic_v6/Chinese_New_Year.json'),user_model=userid)
        return ImageWrapper(result),f"用户id是{userid}"
                


