def test_role():
    role_template = '你扮演一个作图师，以传入的图片制作新年风格图。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = [{'name': 'photo_gen'}]
    from modelscope_agent.agents.role_play import RolePlay  # NOQA
    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)
    response = bot.run('上传文件杨幂_666.jpg')
    text = ''
    for chunk in response:
        text += chunk
    print(text)


# test_role()


# 在执行后面的代码之前，必须先执行上面的函数创建人物LoRA，以及useid.txt，否则无法执行！
def test_infer_tool_name():
    role_template = '你扮演一个作图师，以传入的人物LoRA和用户所需风格制作图片。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = [{'name': 'photo_infer'}]
    from modelscope_agent.agents.role_play import RolePlay  # NOQA
    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)
    response = bot.run('根据训练好的人物LoRA，生成图片，所需要的风格是中国新年风')
    text = ''
    for chunk in response:
        text += chunk
    print(text)


# test_infer_tool_name()


def test_infer_tool_path():
    role_template = '你扮演一个作图师，以传入的人物LoRA和用户所需风格制作图片。'

    llm_config = {'model': 'qwen-max', 'model_server': 'dashscope'}

    # input tool args
    function_list = [{'name': 'photo_infer'}]
    from modelscope_agent.agents.role_play import RolePlay  # NOQA
    bot = RolePlay(
        function_list=function_list, llm=llm_config, instruction=role_template)
    response = bot.run(
        '根据训练好的人物LoRA，生成图片，所需要的风格文件的路径为/home/wsco/modelscope-agent/modelscope_agent/tools/facechain_utils/styles/MajicmixRealistic_v6/Chinese_New_Year.json'
    )
    text = ''
    for chunk in response:
        text += chunk
    print(text)


test_infer_tool_path()
