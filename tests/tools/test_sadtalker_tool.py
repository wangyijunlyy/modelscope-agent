def test_tool():
    import torch
    torch.multiprocessing.set_start_method('spawn')

    from modelscope_agent.tools.sadtalker_tool import SadTalkerTool

    speaker_path = '/home/wsco/wyj/Bert-VITS2/胡桃.png'
    audio_path = '/home/wsco/wyj/Bert-VITS2/output.wav'
    params = f'{{"audio_path": "{audio_path}","speaker_path": "{speaker_path}"}}'
    tool = SadTalkerTool()
    output_path = tool.call(params)
    assert (isinstance(output_path, str))
    print(output_path)


def test_agent():
    import torch
    torch.multiprocessing.set_start_method('spawn')
    from modelscope_agent.prompts.role_play import RolePlay  # NOQA
    from modelscope_agent.tools.sadtalker_tool import SadTalkerTool

    role_template = '你扮演一个说话视频创造者，将说话人图片和音频融合转成视频。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = [{'name': 'sadtalker_tool'}]

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run(
        '说话人图片路径是"/home/wsco/wyj/Bert-VITS2/胡桃.png"，声音文件路径是"/home/wsco/wyj/Bert-VITS2/output.wav"，可以生成个视频吗'
    )

    text = ''
    for chunk in response:
        text += chunk
    assert (isinstance(text, str))
    print(text)


if __name__ == '__main__':
    test_agent()
