# From /data/projects/gorge_walk/kaiwu_agent/gorge_walk/diy/eval_workflow.py
from kaiwu_agent.utils.common_func import attached
from kaiwu_agent.agent.protocol.protocol import observation_process, action_process
import random, os
from kaiwu_agent.conf import yaml_gorge_walk_game as game_conf
from diy.utils.drawer import Drawer
from diy.utils import show_debug

def workflow(envs, agents, logger=None, monitor=None, treasure_cnt=5, treasure_random=False):    
    env, agent = envs[0], agents[0]

    treasure_cnt = game_conf.treasure_num
    treasure_random = game_conf.treasure_random

    list_treasure_id = []
    if treasure_random:
        numbers = list(range(10))
        if treasure_cnt:
            list_treasure_id = sorted(random.sample(numbers, treasure_cnt))
    else:
        if treasure_cnt:
            # list_treasure_id = list(range(treasure_cnt))
            list_treasure_id = [2,5,6,7,9]
    
    logger.info("Start Evaluation ...")
    EPISODE_CNT = 1
    total_score, win_cnt, treasure_cnt = 0, 0, 0
    for episode in range(EPISODE_CNT):
        usr_conf = {"diy": {
            'start': [29, 9],
            'end': [11, 55],
            'treasure_id': list_treasure_id,
            }
        }
        
        obs = env.reset(usr_conf=usr_conf)
        obs_data = observation_process(obs)
        done = False
        drawer = Drawer()
        drawer.update_state(obs)
        drawer.build()
        # logger.info(f"build maze figure")
        while not done:
            act_data = agent.exploit(list_obs_data=[obs_data])[0]
            act = action_process(act_data)
            frame_no, _obs, score, terminated, truncated, game_info = env.step(act)
            obs_data = observation_process(_obs)
            drawer.update_state(_obs)
            drawer.build()
            # logger.info(f"build maze figure")
            done = terminated or truncated
            if terminated:
                win_cnt += 1
            total_score += score
        
        treasure_cnt += game_info.treasure_count
        if episode % 10 == 0 and episode > 0:
            logger.info(f"Episode: {episode + 1} ...")
            
    logger.info(f"Average Total Score: {total_score / EPISODE_CNT}")
    logger.info(f"Average Treasure Collected: {treasure_cnt / EPISODE_CNT}")
    logger.info(f"Success Rate : {win_cnt / EPISODE_CNT}")