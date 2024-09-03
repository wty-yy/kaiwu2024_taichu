from kaiwu_agent.utils.common_func import attached

@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    N = int(1e5+100)
    for i in range(N):
        agent.learn([None])  # 保存 id = 10000 * n
    agent.save_model()  # 保存 id = 100100
    return
