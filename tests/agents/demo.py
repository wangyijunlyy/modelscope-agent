import os
from concurrent.futures import ProcessPoolExecutor

from modelscope_agent.agents.role_play import RolePlay

if __name__ == '__main__':
    # 定义指令
    role_template_photo = '你扮演一个写真照生成小助手，你需要根据用户上传的照片生成新年风格的写真照。'
    role_template_voice = '你扮演一个声音克隆生成小助手，你需要先根据用户上传的音频训练，然后根据用户想要的新年祝福语生成对应的音频。'
    role_template_video = '你扮演一个视频生成小助手，你需要根据前面两个助手生成的写真照和音频，生成一段新年祝福视频。'
    # 定义参数
    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    function_list_photo = [{'name': 'photo_gen'}]
    function_list_voice = [{'name': 'voice_gen'}]
    function_list_video = [{'name': 'video_gen'}]

    prompt_photo = '[上传文件杨幂_666.jpg]'
    prompt_voice = '[上传音频test_dongxuelian_888.wav，新的一年即将到来，恭喜发财！]'  # 这里的祝福语应该依赖agnet结合知识库生成

    # 定义agent
    bot_photo = RolePlay(
        function_list=function_list_photo,
        llm=llm_config,
        instruction=role_template_photo)
    bot_voice = RolePlay(
        function_list=function_list_voice,
        llm=llm_config,
        instruction=role_template_voice)
    bot_video = RolePlay(
        function_list=function_list_video,
        llm=llm_config,
        instruction=role_template_video)
    response_image = bot_photo.run(prompt_photo)
    response_voice = bot_voice.run(prompt_voice)
    text = ''
    for chunk in response_image:
        text += chunk
    for chunk in response_voice:
        text += chunk
    result = bot_video.run(text)
    print(result)
