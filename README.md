# Conda 安装说明：RTX 3090 / Driver 560.35.05

修订日期：2026-09-17。适用于已经上传并解压旧 v2 文件夹的用户。

## 本次改动与验证范围

不再使用 uv，也不创建 `.venv-serve` / `.venv-eval`。改用两个 Conda 环境：

| 环境 | 名称 | 负责内容 |
|---|---|---|
| 服务端 | `kvarn-serve` | 固定 KVarN fork、PyTorch、TurboQuant/KVarN 推理 |
| 客户端 | `evalscope-client` | EvalScope，通过 HTTP 测评 |

Conda 负责创建/激活 Python 环境；固定的 Torch CUDA wheel、KVarN 源码包和 EvalScope 仍使用**激活环境内的 `python -m pip`**安装。不在 base 中安装它们，不交替使用 Conda/Pip 升级同一套 GPU 依赖，不使用 `pip --user` 或 `sudo pip`。

本次沿用 v2 的版本选择：Python 3.12.13、CUDA Toolkit 12.6.3、Torch 2.11.0+cu126、固定 KVarN commit、EvalScope 1.12.0。版本依据保留在 VERSION_NOTES.md；本次不是对 GPU 兼容性的新实机认证。

已修正旧脚本中的 venv 激活路径与 preflight 环境判断。新增的检查使用 CONDA_PREFIX、conda-meta、实际解释器和环境名，不把 `sys.prefix == sys.base_prefix` 当作 Conda 未激活的依据。

验证范围见 VALIDATION.txt：静态检查与不依赖 GPU 的逻辑测试，不包含在目标机创建 Conda 环境、安装全部依赖、编译 CUDA 或模型推理。

## 0. 目录已经上传：不再下载或解压旧实验包

本说明假设旧目录位于 `/root/kv_evalscope_3090_cu126_v2`。`/root/` 是 root 用户家目录，不是系统根目录 `/`。如果实际上传到了 `/kv_evalscope_3090_cu126_v2`，只把 KIT 改为这个路径；不要移动已有模型、源码或结果。

```bash
export KIT=/root/kv_evalscope_3090_cu126_v2
export KVBENCH_ROOT=/root/kvbench-3090-cu126
export ROOT="$KVBENCH_ROOT"

ls "$KIT/install_serve.sh" "$KIT/install_eval.sh" "$KIT/preflight.py"
mkdir -p "$ROOT"/{src,toolchains,downloads,logs,locks,constraints,cache,results}
```

KIT 是脚本目录；ROOT 是安装源码、缓存和保存结果的工作目录，不要求把脚本搬到 ROOT。采用完整 v3 包而不是补丁时，把 KIT 指向实际解压的 v3 目录，后续步骤一致。

### 对旧 v2 目录应用补丁

只需把 `update_kvbench_to_conda.py` 上传到 `/root/`。这个单文件补丁已包含更新后的脚本和说明，不再下载其他文件、不连接 GPU、不安装包。

用现有 Python 3.8+ 运行即可，**补丁不要求已创建实验 Conda 环境**：

```bash
python --version
python /root/update_kvbench_to_conda.py --kit "$KIT"
```

`python` 不存在但 `python3` 可用时，以 `python3` 运行同一命令。两者都没有时，先在第 1 步初始化已有 Conda，然后用 Conda base 的 Python 运行补丁；这只是改文件，不在 base 安装依赖。

补丁先校验旧文件，发现本地改动会停止，避免覆盖。它只更新明确列出的安装/检查/启动脚本和说明，自动备份到 `KIT/_backup_before_conda_时间戳/`。不删除旧环境、不修改 src、模型、数据或 results。再次运行已完成的补丁不会重复修改。

```bash
ls "$KIT/check_conda_env.py"
head -n 1 "$KIT/README.zh-CN.md"
```

不要只把 README 改成 Conda 后继续运行旧安装脚本；旧脚本仍会寻找 `.venv-*`。

## 1. 初始化已有 Conda

```bash
conda --version
conda info --base
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list
```

这在当前 Bash 中加载激活函数，不要求重装 Conda，也不要求执行 `conda init` 修改全局 shell 配置。

若提示 `conda: command not found`，但已经装有 Conda，先确认真实安装目录。例如确实存在 `/root/miniconda3/etc/profile.d/conda.sh` 时：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda --version
```

其他常见路径是 `/opt/conda`、`/root/anaconda3`；使用实际存在的目录，不盲目 source 一个猜测路径。本说明不包含在未安装 Conda 的机器上重新安装发行版。

## 2. 创建两个 Conda 环境

先检查 `conda env list`，不要删除已有同名环境来重建。下面针对尚未创建这两个环境的情况。

```bash
# 检查所选 channel 是否提供目标 Python 补丁版本。
conda search --override-channels -c conda-forge 'python=3.12.13'

conda create -y --no-default-packages \
  --override-channels -c conda-forge \
  -n kvarn-serve python=3.12.13 pip

conda create -y --no-default-packages \
  --override-channels -c conda-forge \
  -n evalscope-client python=3.12.13 pip
```

公司使用内部镜像时，两个命令统一改用批准的 channel，不关闭 SSL 检查。此处不声称你的镜像已同步 3.12.13。如果返回 PackagesNotFoundError，先检查实际可用的 3.12.x；需更换补丁版本时，两边选同一版本，并保存实际版本。核心预检要求 Python 3.12，不要求只有 3.12.13 才能运行。

```bash
conda activate kvarn-serve
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

python --version
which python
python -m pip --version
echo "$CONDA_PREFIX"
python "$KIT/check_conda_env.py" --expected-name kvarn-serve
```

终端一般显示 `(kvarn-serve)`，且 Python / pip 的路径都应属于该环境。如果以前激活过 Python venv，先退出旧 venv，再激活 Conda，不要嵌套。

本轮没有必要另装 uv、系统 python3.12 或再套一层 Python venv。

## 3. 检查主机工具链：已经满足就跳过安装

以下主机包命令假设 Ubuntu 22.04/24.04、Linux x86_64。其他发行版需要对应包管理命令。

```bash
cat /etc/os-release
uname -m
ldd --version | head -n 1
nvidia-smi | tee "$ROOT/logs/nvidia-smi.before.txt"
command -v nvcc || true
nvcc --version || true
gcc-12 --version
g++-12 --version
```

缺少编译依赖时再安装；已经是 root 时不需要 sudo：

```bash
apt-get update
apt-get install -y \
  build-essential gcc-12 g++-12 git curl wget \
  pkg-config libssl-dev libnuma-dev libclang-dev perl protobuf-compiler
```

普通用户使用 sudo；没有权限时由管理员提供工具。本方案不安装 `nvidia-driver-*`，不修改你现有 Driver 560.35.05。

Conda 环境与 CUDA Toolkit 是不同层次：更换环境管理方式不代表要重装已经正确配置的 Toolkit。`nvidia-smi` 显示的 CUDA Version 也不是本地 nvcc 安装证明。

### 已有完整 CUDA Toolkit 12.6

```bash
export CUDA_HOME=/usr/local/cuda-12.6
"$CUDA_HOME/bin/nvcc" --version
"$CUDA_HOME/bin/ptxas" --version
```

如果之前按 v2 安装在工作目录中，使用原路径：

```bash
export CUDA_HOME="$ROOT/toolchains/cuda-12.6"
```

二者择一，不重复安装。两项工具都应报告 CUDA 12.6。

### 确实没有 Toolkit 12.6 时

沿用 v2 的 Toolkit-only 安装路径，不安装 runfile 内的驱动：

```bash
cd "$ROOT/downloads"
wget -c https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
mkdir -p "$ROOT/toolchains/cuda-12.6" "$ROOT/cache/cuda-installer-tmp"

sh cuda_12.6.3_560.35.05_linux.run \
  --silent --toolkit \
  --toolkitpath="$ROOT/toolchains/cuda-12.6" \
  --defaultroot="$ROOT/toolchains/cuda-12.6" \
  --tmpdir="$ROOT/cache/cuda-installer-tmp"

export CUDA_HOME="$ROOT/toolchains/cuda-12.6"
```

不加 `--driver`；遇到权限/依赖错误先处理，不无条件使用 `--override`。不在此方案中额外 `conda install pytorch-cuda` 或 `cudatoolkit` 覆盖所选 GPU 依赖链。

### Rust 1.95

已有该版本时直接检查：

```bash
export PATH="$PATH:$HOME/.cargo/bin"
rustc +1.95 --version
cargo +1.95 --version
```

已有 rustup、但没有该版本时：

```bash
rustup toolchain install 1.95 --profile minimal
```

确实没有 rustup 时，才执行：

```bash
curl --fail --location https://sh.rustup.rs -o "$ROOT/downloads/rustup-init.sh"
sh "$ROOT/downloads/rustup-init.sh" \
  -y --profile minimal --default-toolchain none --no-modify-path
export PATH="$PATH:$HOME/.cargo/bin"
rustup toolchain install 1.95 --profile minimal
```

### 执行预检

```bash
conda activate kvarn-serve
unset PYTHONHOME PYTHONPATH
source "$KIT/env.sh"
python "$KIT/preflight.py"
```

preflight 未通过时先停止，不继续编译。

## 4. 安装服务端：在 kvarn-serve 内运行

```bash
conda activate kvarn-serve
unset PYTHONHOME PYTHONPATH
source "$KIT/env.sh"

CUDA_VISIBLE_DEVICES=0 bash "$KIT/install_serve.sh"
```

更新后的脚本**不自动激活其他环境**，只使用当前 Conda 解释器。脚本安装 cu126 Torch、固定源代码及构建依赖，执行 PyTorch/Triton 检查，再源码编译 KVarN fork。不要额外重复执行旧的 pip 安装命令。

固定 KVarN commit：`7586257f1c632e63187bfacbbe21ccb51540f7b3`。

核心依赖仍为 Torch 2.11.0+cu126、torchvision 0.26.0+cu126、torchaudio 2.11.0+cu126，使用源码构建的 `--no-build-isolation`。本次不变更这些版本，也不重新宣称它们已经在你的 GPU 上验证。

```bash
python -m pip check
python - <<'PY'
import sys
import torch
import vllm
print('Python:', sys.executable)
print('Torch:', torch.__version__)
print('Torch CUDA:', torch.version.cuda)
print('vLLM source:', vllm.__file__)
print('CUDA available:', torch.cuda.is_available())
PY
```

检查目标：Python 属于 kvarn-serve，Torch 为 2.11.0+cu126，CUDA 构建为 12.6，vLLM 路径位于 `$ROOT/src/KVarN/vllm/`。

不要在这个环境执行 `pip install -U vllm`、`conda update --all`，也不随意升级 Torch/Triton/FlashInfer。编译失败应保留 `logs/build-kvarn-cu126.log`，不是通过随机改依赖尝试绕过。

## 5. 安装客户端：先切换到 evalscope-client

必须等服务端安装成功并生成 `constraints/tokenizer-stack.txt` 后再执行：

```bash
conda activate evalscope-client
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

python "$KIT/check_conda_env.py" --expected-name evalscope-client
bash "$KIT/install_eval.sh"

python -c 'import importlib.metadata as m; print(m.version("evalscope"))'
evalscope perf --help
```

版本仍使用 EvalScope 1.12.0；脚本将服务端导出的 tokenizer 版本作为约束。客户端不主动安装另一份 vLLM 或 GPU Torch。

## 6. 模型已有则直接复用

已有本地固定模型 snapshot：

```bash
export MODEL=/absolute/path/to/Qwen3-4B-snapshot
printf 'export MODEL=%q\n' "$MODEL" > "$ROOT/model.env"
```

没有模型才下载：

```bash
conda activate evalscope-client
export HF_HOME="$ROOT/cache/huggingface"
python "$KIT/download_model.py"
source "$ROOT/model.env"
```

已经保存 model.env 时，只需 `source "$ROOT/model.env"`，无需再次下载。服务端与客户端都使用这份路径。

## 7. 两个终端进行首次测试

### 终端 A：启动服务端

```bash
export KIT=/home/wanglin/kv_evalscope_3090_cu126_v2
export KVBENCH_ROOT=/home/wanglin/kvbench-3090-cu126
export ROOT="$KVBENCH_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kvarn-serve
unset PYTHONHOME PYTHONPATH
source "$KIT/env.sh"
source "$ROOT/model.env"
cd "$KIT"

CUDA_VISIBLE_DEVICES=0 MAXLEN=4096 MAXSEQ=1 APC=0 \
  bash serve.sh fp16 --enforce-eager
```

非标准 Toolkit 路径且未被自动发现时，在 `source env.sh` 前重新 export CUDA_HOME。`conda activate` 不会自动加载这个项目的 env.sh。

### 终端 B：客户端

```bash
export KIT=/home/wanglin/kv_evalscope_3090_cu126_v2
export KVBENCH_ROOT=/home/wanglin/kvbench-3090-cu126
export ROOT="$KVBENCH_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate evalscope-client
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
source "$ROOT/model.env"
cd "$KIT"

curl --fail http://127.0.0.1:8000/v1/models

LENGTHS='1024' CONCURRENCY='1' NUMBER=8 OUTPUT_TOKENS=256 \
  bash bench.sh fixed fp16
```

FP16 完成后，在终端 A 按 Ctrl-C 停止该服务，确认它释放资源，再依次将服务命令方法名替换为 `tq4`、`kvarn`。客户端标签同步改为相同方法。不要同时启动三个服务，不要杀其他人的进程。

```bash
# 终端 A，分开执行：
CUDA_VISIBLE_DEVICES=0 MAXLEN=4096 MAXSEQ=1 APC=0 bash serve.sh tq4 --enforce-eager
CUDA_VISIBLE_DEVICES=0 MAXLEN=4096 MAXSEQ=1 APC=0 bash serve.sh kvarn --enforce-eager

# 终端 B，每次与实际服务对应，只执行当前方法：
LENGTHS='1024' CONCURRENCY='1' NUMBER=8 OUTPUT_TOKENS=256 bash bench.sh fixed tq4
LENGTHS='1024' CONCURRENCY='1' NUMBER=8 OUTPUT_TOKENS=256 bash bench.sh fixed kvarn
```

`bench.sh` 方法名只是结果标签，不切换服务器。三种方法通过后，再沿用 PROTOCOL.md 扩展并发、长上下文、前缀缓存和自然生成质量测评。

## 8. 日后重新登录：无需重新安装

服务端每次使用：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kvarn-serve
export KVBENCH_ROOT=/home/wanglin/kvbench-3090-cu126
export KIT=/home/wanglin/kv_evalscope_3090_cu126_v2
unset PYTHONHOME PYTHONPATH
source "$KIT/env.sh"
source "$KVBENCH_ROOT/model.env"
```

客户端每次使用：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate evalscope-client
export KVBENCH_ROOT=/home/wanglin/kvbench-3090-cu126
export KIT=/home/wanglin/kv_evalscope_3090_cu126_v2
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
source "$KVBENCH_ROOT/model.env"
```

不用再 source `.venv-*/bin/activate`；也不要 source 某个猜测的 Conda 环境 `bin/activate` 来替代 `conda activate`。

## 9. 文件记录与故障定位

安装完成后生成：

```text
locks/serve.conda.yml
locks/serve.conda-explicit.txt
locks/serve.freeze.txt
locks/serve.inspect.json
locks/eval.conda.yml
locks/eval.conda-explicit.txt
locks/eval.freeze.txt
locks/eval.inspect.json
locks/kvarn.commit.txt
```

Conda explicit 记录 Conda 包，不等于完整的 pip/CUDA 扩展锁。保存两套记录、源代码 commit、模型 revision 和驱动/GPU 信息；不要声称单个 YAML 能无条件跨硬件复现所有二进制。

| 提示 | 应对 |
|---|---|
| 仍要求创建 `.venv-serve` | 补丁未应用到实际 KIT，或仍运行另一目录中的旧脚本 |
| 环境不匹配 | `conda activate kvarn-serve` 或 `conda activate evalscope-client`，不让脚本偷偷切换 |
| CONDA_PREFIX 与 Python 不一致 | 检查 `type -a python`、旧 venv、Python alias、PATH |
| base 环境被拒绝 | 新建并激活独立实验环境，不在 base 安装 GPU 栈 |
| PYTHONHOME / PYTHONPATH | 本实验终端 unset 后重新检查，避免导入旧包 |
| 缺少 tokenizer-stack.txt | 先完成服务端安装，不能先安装不对齐的客户端依赖 |
| CUDA / 编译 / 推理错误 | 环境迁移不代替 GPU 兼容性验证，保存完整日志按层排查 |

## 官方参考

- Conda 创建与管理环境：https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html
- Conda 激活机制：https://docs.conda.io/projects/conda/en/stable/dev-guide/deep-dives/activation.html
- Conda/Pip 包管理：https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-pkgs.html
- Conda 导出：https://docs.conda.io/projects/conda/en/stable/commands/env/export.html
- Python venv 的 prefix 语义：https://docs.python.org/3.12/library/venv.html

GPU 栈的原版本依据见 VERSION_NOTES.md。
