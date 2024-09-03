# 2024开悟比赛

> [比赛代码逻辑笔记](./assets/code_logic.md)

## 环境启动方法
> 为了能直接使用 `test_train.py` 进行模型训练并保存（比客户端训练效率更高），显示当前镜像下载速度，推荐如下直接使用命令行打开镜像的方法。

我们以打开走迷宫的代码为例：

1. 在腾讯开悟APP中设置工作空间为 `.../kaiwu2024/gorgewalk/`（路径自选），**启动开发环境**（CPU或GPU随便选），在**容器检测已完成**后就可以停止启动。

2. 打开 `.../kaiwu2024/gorgewalk/dev/.docker-compose.yaml` 文件，找到容器 `kaiwudrl` 中的 `volumnes` 项（第34行，版本4.2.9），向其中加入如下信息
```yaml
- '${KAIWU_CODE_FILE}/../train:/workspace/train'
```

3. 打开终端输入
```shell
docker compose -f {你的路径}/kaiwu2024/gorgewalk/dev/.docker-compose.yaml -p kaiwu-dev up -d
```

4. 你可以用网络版的 vscode，由于体验太差，本地 vscode 可以通过安装 `Remote Explorer` 和 `Dev Containers`，点击左下角的小按钮（打开远程窗口），选择**附加到正在运行的容器**，选择 `.../win_gpu/...` 即可进入容器，打开文件夹 `/data/projects/gorge_walk/` 即可进入相同的工作路径下。

5. 安装 vscode 插件，推荐安装 `Python` 和 `Even Better TOML` 两个插件。

## 注意事项
### diy决策模型编写
只需要重写 `diy/algorithm/agent.py` 中的 `exploit, save_model, load_model` 三个函数，其余的 `predict, learn` 都可以 `pass` 掉（默认的 `wrapper` 还是都要的）。

`save_model, load_model` 的保存 `id` 就是 `learn` 函数的调用次数，**虽然我们 pass 掉了 `learn` 函数，但是后端会自动记录 `learn` 函数调用次数，因此我们需要在每次更新模型时候调用一次 `agent.learn()`**，且模型会自动在 `id` 为 `10000` 的倍数时，自动调用 `agent.save_model`。
### 模型训练方法（无需使用客户端）
写好你的 `diy` 文件夹后，**检查 `config/configure_app.toml` 文件的最后两行，如果有 `eval_model_cir, eval_model_id` 属性，请删去（否则会出现重复配置，从而报错）**，直接运行 `train_test.py` 等待训练完成即可，因为我们之前装载了 `/workspace/train` 文件夹，所以直接在本机的工作路径下的 `train/` 文件夹中，就能看到压缩好的待上传文件
> 如果你没看到保存的文件，请到 docker 中的 `/data/ckpt/gorge_walk_diy/` 中，看你的模型是否有保存，如果显示保存 `id=0`，则说明你忘记在更新模型时候调用一次 `agent.learn()`。
> 
> 如果发现有存储的模型参数，那么请按照[模型上传方法](#模型上传方法)中的压缩文件格式，自己创建一个压缩包。
> 
### 本地模型验证方法（检测模型是否读取成功）
1. 将你要读取的模型权重文件 `model.ckpt-[id].*` 放到 `ckpt/` 文件夹下（确保和 `agent.load_model` 的文件格式一致）

2. 向 `config/configure_app.toml` 文件最后加入如下两行（如果存在这两行配置，请检查模型编号是否正确）：
```toml
eval_model_dir = "/data/projects/gorge_walk/ckpt"  # 模型存储的文件夹位置（无需修改）
eval_model_id = "1874992"  # 修改为你模型文件名中的id编号
```

将 `config/configure_app.toml` 中的 47 行 `run_mode` 从 `train` 改为 `eval`，运行 `train_test.py` 即可自动调用 `agent.load_model` 传入参数 `path=eval_model_dir, id=eval_model_id`，从而读取模型权重，再传入到 `kaiwu_agent/gorge_walk/diy/eval_workflow.py` 的 `workflow(...)` 函数中进行验证。
> 如果想要自己在验证中输出模型信息，请直接对 `eval_workflow.py` 进行修改，需要注意 docker 每次重启就会重置该文件。
### 模型上传方法
执行 `train_test.py` 后，会自动将模型权重保存在 `/workspace/tarin/backup_model/` 下，其中只需包含三个文件夹，`ckpt/, conf/, diy/`：
- `diy/`：就是工作路径下的 `code/diy`（diy 为算法名称，可以自行替换为那5个名字之一）
- `conf/`：就是工作路径下的 `code/conf/`
- `ckpt/`：只需包含模型参数文件，可以从 `/data/ckpt/gorge_walk_diy/` 中拷贝过来的，注意要和 `config/configure_app.toml` 中最后一行的 `eval_model_id` 编号保持一致。
### 新增模型后检测失败自检方法
1. 检查压缩包中的 `config/configure_app.toml` 的最后几行，变量有没有重复定义，最后两行应该是如下
```toml
eval_model_dir = "/data/projects/gorge_walk/ckpt"
eval_model_id = "1234"  # 你的模型编号
```

2. 按照[本地模型验证方法](#本地模型验证方法检测模型是否读取成功)，看你的本地是否可以启动模型验证过程，本地验证失败可以本地DEBUG，而传上去失败了，什么日志都不会有。
### 训练无法终止问题
当执行完 `train_test.py` 后，直接 `Ctrl+C` 可能存在大量还没终止的进程，占用大量CPU和内存，我们可以再开一个终端，分别执行 `pkill bash` 和 `pkill python3` 将他们杀死。（还有些关不掉的不清楚什么原因）

## 	海选赛-走迷宫
（可选，没有 `wandb` 则不会上传）支持 wandb 自动同步 Tensorboard 日志，进入容器后，需要先用 `pip install wandb` 再用 `wandb login` 输入你的 token 登录账号，即可自动上传（修改 `agent.py` 中的62行组织名称，如果没有组织则请注释掉）。

### 注意事项
环境返回的状态可以划分为如下结构（宝箱用 `star` 表示）：
```python
def state2dict(s: list):  # 环境返回的长度为250的状态数组
  assert len(s) == 250
  s = np.array(s, np.float32)
  s = {
    # s[0]: 位置的常量表示的形式 x*64+y
    'pos': (np.argmax(s[1:65]), np.argmax(s[65:129])),  # 当前位置 (x, y)
    'dist': {
      'end': s[129],  # 距离终点距离0~6
      'star': s[130:140],  # 距离宝箱距离0~6, 当宝箱不存在时为999
    },
    'map': {  # 以当前位置为中心，周围的5x5范围区域
      'wall': s[140:165].reshape(5, 5),  # 墙为1，通路为0
      'star': s[165:190].reshape(5, 5),  # 宝箱为1
      'end': s[190:215].reshape(5, 5),  # 终点为1
    },
    'memory': s[215:240].reshape(5, 5),  # 记录走过的次数*0.1（撞墙不会增加当前位置的次数）
    'star_flag': s[240:250]  # 宝箱是否存在0/1标志
  }
  types = {
    'pos': np.int32,
    'dist': np.int32,
    'map': bool,
    'memory': np.float32,
    'star_flag': bool 
  }
  for k, v in s.items():
    t = types[k]
    if not isinstance(v, dict):
      s[k] = np.array(v, t)
    else:
      for kk in v:
        v[kk] = np.array(v[kk], types[k])
  return s
```
