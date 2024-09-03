from kaiwu_agent.agent.base_agent import BaseAgent
from kaiwu_agent.agent.base_agent import (
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    predict_wrapper,
    exploit_wrapper,
)
from diy.feature.definition import ActData
from diy.utils import ProcessPrinter
import numpy as np

class Agent(BaseAgent):

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)
        self.logger, self.moniter = logger, monitor
        self.process_printer = ProcessPrinter()
        self.logger.info("Init Agent")
        self.process_printer("Init Agent")
        self.cnt = 0

    @predict_wrapper
    def predict(self, list_obs_data):
        # 这里智能体预测数组长度必定为1个，是直接从train_workflow.py中调用的（应该是aisvr进程）
        assert len(list_obs_data) == 1, self.process_printer(f"[ERROR] {len(list_obs_data)=}, WHY?")
        return [ActData(act=1)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return [ActData(act=1)]

    @learn_wrapper
    def learn(self, list_sample_data):
        self.cnt += 1
        # self.process_printer(f"{self.cnt=}, {list_sample_data=}, {len(list_sample_data)=}")
        self.logger.info(f"{self.cnt=}, {list_sample_data=}, {len(list_sample_data)=}")

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        np.save(model_file_path, [1,2,3])
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"): ...
