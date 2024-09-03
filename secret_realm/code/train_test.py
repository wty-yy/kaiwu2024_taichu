#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :train_test.py
@Author  :kaiwu
@Date    :2022/10/20 11:43

"""

import time
import os
import platform
from multiprocessing import Process
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.http_utils import http_utils_request
from kaiwudrl.common.utils.common_func import (
    python_exec_shell,
    find_pids_by_cmdline,
    scan_for_errors,
)
import kaiwudrl.server.learner.learner as learner
import kaiwudrl.server.aisrv.aisrv as aisrv
from kaiwudrl.common.config.config_control import CONFIG
from typing import List

# To run the train_test, you must modify the algorithm name here. It must be one of dqn, target_dqn, diy.
# Simply modify the value of the algorithm_name variable.
# 运行train_test前必须修改这里的算法名字, 必须是dqn、target_dqn、diy里的一个, 修改algorithm_name的值即可
algorithm_name_list = ["dqn", "target_dqn", "diy"]
algorithm_name = "diy"


# train
# 训练
def train():

    start_time = time.time()

    # To modify the value in the environment variable and initiate training for the learner as soon as possible
    # 修改环境变量里的值, 尽快让learner进行训练
    os.environ.update(
        {
            "use_ckpt_sync": "False",
            "wrapper_type": "local",
            # "replay_buffer_capacity": "256",
            # "preload_ratio": "1",
            # "train_batch_size": "16",
            # "use_prometheus": "True",
            # "aisrv_connect_to_kaiwu_env_count": "1",
        }
    )

    # To modify the configuration and execute the script directly
    # 修改配置, 调用脚本直接执行
    if algorithm_name not in algorithm_name_list:
        print("\033[92m" + f"algorithm_name: {algorithm_name} not in list {algorithm_name_list}" + "\033[0m")

        python_exec_shell(f"sh tools/stop.sh all")

    python_exec_shell(f"sh /root/tools/change_algorithm_all.sh {algorithm_name}")
    print(f"current algorithm_name is {algorithm_name}")

    # Setting the sample transmission to either Reverb or ZMQ based on the operating system
    # 根据不同的操作系统设置发送样本的是reverb还是zmq
    architecture = platform.machine()
    platform_maps = {
        "aarch64": "zmq",
        "arm64": "zmq",
        "x86_64": "reverb",
        "AMD64": "reverb",
    }
    sample_tool_type = platform_maps.get(architecture)
    if sample_tool_type is None:
        print(f"Architecture '{architecture}' may not exist or not be supported.")
    else:
        result_code, result_str = python_exec_shell(f"sh tools/change_sample_server.sh {sample_tool_type}")
        if result_code != 0:
            raise ValueError(f"Execution error! Please check the error detail: {result_str}")

    CONFIG.set_configure_file("conf/kaiwudrl/learner.toml")
    CONFIG.parse_learner_configure()

    # To delete previous model files and logs
    # 删除以前的model文件和日志
    python_exec_shell(f"rm -rf {CONFIG.user_ckpt_dir}/{CONFIG.app}_{algorithm_name}/*")
    python_exec_shell(f"rm -rf {CONFIG.log_dir}/*")
    done_file = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/process_stop.done"
    if os.path.exists(done_file):
        os.remove(done_file)

    # To start aisrv, learner, actor, and battlesrv
    # 启动aisrv, learner, actor, battlesrv
    procs: List[Process] = []
    procs.append(Process(target=learner.main, name="learner"))
    procs.append(Process(target=aisrv.main, name="aisrv"))

    for proc in procs:
        proc.start()
        time.sleep(10)
        check(proc)

    # while True:

    #     # the method of obtaining monitoring values is adopted
    #     # 采用获取监控值的方法
    #     success = check_train_success_by_monitor()
    #     if success:

    #         time.sleep(5)
    #         print(
    #             "\033[1;32m"
    #             + "Train test succeeded, will exit, cost "
    #             + f"{time.time() - start_time:.2f} seconds"
    #             + "\033[0m"
    #         )
    #         python_exec_shell("sh tools/stop.sh all")

    #     # time.sleep(1)
    #     # for proc in procs:
    #     #     check(proc)

def check_process_stop_done():
    done_file = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/process_stop.done"
    if os.path.exists(done_file):

        time.sleep(5)
        print("\033[1;31m" + "find process_stop.done file, so exit" + "\033[0m")
        os.remove(done_file)
        python_exec_shell(f"sh tools/stop.sh all")

# To check if a process is alive, any error log
# 检测进程是否存活, 是否有错误日志
def check(proc: Process):
    if proc.is_alive():
        print(f"{proc.name} is alive")

        # If an error log is generated, exit early
        # 如果有错误日志产生, 提前退出
        if scan_for_errors(CONFIG.log_dir, error_indicator="ERROR"):

            time.sleep(5)
            print("\033[1;31m" + "find error log, please check" + "\033[0m")
            python_exec_shell(f"sh tools/stop.sh all")

        # If the file 'process_stop.done' is created, exit prematurely
        # 如果有process_stop.done的文件生成, 提前退出
        check_process_stop_done()

    else:

        time.sleep(5)
        print("\033[1;31m" + f"{proc.name} is not alive, please check error log" + "\033[0m")
        python_exec_shell(f"sh tools/stop.sh all")

# Determine the success of training based on the reported monitoring values.
# If the number of training steps is greater than 0,
# it indicates a successful training
# 按照上报监控的值来判断训练是否成功, 如果训练步数大于0则代表训练成功
def check_train_success_by_monitor():
    try:
        pushgateway = os.environ.get("prometheus_pushgateway")
        url = f"http://{pushgateway}/api/v1/metrics"
        resp = http_utils_request(url)
        if not resp:
            return False

        datas = resp.get("data", [])
        pids = find_pids_by_cmdline("train_test")
        pids = [str(pid) for pid in pids]

        for data in datas:
            if "train_global_step" not in data:
                continue

            train_global_step = data.get("train_global_step", {})
            metrics = train_global_step.get("metrics", [])
            if process_monitor_metrics(metrics, pids):
                return True

        return False

    except Exception:
        return False


# Processing monitor metrics
# 处理监控指标
def process_monitor_metrics(metrics: List, pids: List[str]):
    for metric in metrics:
        labels = metric.get("labels", {})
        value = metric.get("value", 0)

        job = labels.get("job", None)
        if job:
            for pid in pids:
                if pid in job and int(value) > 0:
                    return True

    return False


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\033[1;31m" + "KeyboardInterrupt, please check" + "\033[0m")
        python_exec_shell(f"sh tools/stop.sh all")
