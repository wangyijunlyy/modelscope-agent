from modelscope_agent.tools.voice_clone import VoiceCloneTool

from modelscope_agent.agents.role_play import RolePlay  # NOQA


def test_voice_clone():
    params = {'audio_path': 'ym_1.wav', 'text': '大家好，我是杨幂，欢迎来到魔搭社区'}
    vc = VoiceCloneTool()
    res = vc.call(params)
    print(res)
    assert (isinstance(res, str))


def test_voice_clone_role():
    role_template = '你扮演一个音频生成工具，根据用户提供的信息调用音频工具生成音频。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = [{'name': 'voice_clone_tool'}]

    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)

    response = bot.run('')

    text = ''
    for chunk in response:
        text += chunk
    print(text)


test_voice_clone()
