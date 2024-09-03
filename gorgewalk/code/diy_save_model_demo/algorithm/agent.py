import numpy as np
from kaiwu_agent.agent.base_agent import BaseAgent
from kaiwu_agent.agent.base_agent import (
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    predict_wrapper,
    exploit_wrapper,
)

class Agent(BaseAgent):

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)
        self.logger = logger

    @predict_wrapper
    def predict(self, list_obs_data): ...

    @exploit_wrapper
    def exploit(self, list_obs_data): ...

    @learn_wrapper
    def learn(self, list_sample_data): ...

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        np.save(model_file_path, [1,2,3])
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"): ...
