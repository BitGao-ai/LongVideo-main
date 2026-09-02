"""分布式训练的分片信息与进程组初始化（DDP 正确性的单一事实源）。

**为什么要单独一个模块**：数据分片（DataLoader 侧）与进程组初始化（Trainer 侧）发生在
*不同时刻*，但两者必须看到同一个 (world_size, rank)。此前 `loaders.py` 只认
`dist.is_initialized()`，而进程组要到 `Trainer.__init__` 才建起来——训练脚本恰恰是
**先建 loader 再建 Trainer**，于是 8 个 rank 在建 loader 时全部拿到 world=1/rank=0，
各自迭代全量数据。配上长度分桶固定 seed，8 张卡算出的是**逐位相同的梯度**，
allreduce 平均 8 份相同值：不崩、不报错、日志正常，但 8 卡等于 1 卡。

解法不是把 `init_process_group` 提到脚本最前面——那会在 fork DataLoader worker 之前
创建 CUDA 上下文，破坏训练脚本刻意安排的"CUDA 初始化前先派生 worker"顺序。
这里改成：**分片信息优先取活跃进程组，取不到就回退读 torchrun 注入的环境变量**。
loader 因此能在进程组建立之前就正确分片，而 CUDA 上下文仍然晚到 Trainer 才创建。

单卡（无 `WORLD_SIZE` 或 `WORLD_SIZE=1`）时本模块全部退化为 (world=1, rank=0)，
与改动前逐行等价，不影响单卡调试。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

__all__ = [
    "env_world_size", "env_rank", "env_local_rank", "launched_distributed",
    "process_group_ready", "dist_info", "get_world_size", "get_rank",
    "is_main_process", "resolve_device", "maybe_init_distributed",
    "check_ddp_launch", "check_device_binding", "add_ddp_args", "resolve_ddp",
    "barrier",
]


# --------------------------- 环境变量（torchrun 注入）---------------------------
def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_world_size() -> int:
    return max(1, _env_int("WORLD_SIZE", 1))


def env_rank() -> int:
    return _env_int("RANK", 0)


def env_local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def launched_distributed() -> bool:
    """是否由 torchrun/torch.distributed.launch 以多进程方式拉起（不代表进程组已建立）。"""
    return env_world_size() > 1


# --------------------------- 分片信息 ---------------------------
def process_group_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_info() -> tuple[int, int]:
    """返回 (world_size, rank)：活跃进程组优先，否则回退环境变量，最后 (1, 0)。

    这是 DataLoader 分片唯一该用的入口——它在进程组建立**之前**也能给出正确答案。
    """
    if process_group_ready():
        return dist.get_world_size(), dist.get_rank()
    if launched_distributed():
        return env_world_size(), env_rank()
    return 1, 0


def get_world_size() -> int:
    return dist_info()[0]


def get_rank() -> int:
    return dist_info()[1]


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    """跨 rank 对齐。NCCL 下显式给 device_ids，否则 PyTorch 会按 current device 猜，
    在没绑好卡的进程里可能卡住（并只打一句 warning）。"""
    if not process_group_ready():
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[env_local_rank()])
    else:
        dist.barrier()


# --------------------------- 设备绑定与进程组 ---------------------------
def _explicit_cuda_index(device: str) -> int | None:
    """从 "cuda:3" 取出 3；不是带显式序号的 cuda 设备则返回 None。"""
    if not device.startswith("cuda") or ":" not in device:
        return None
    try:
        return int(device.split(":", 1)[1])
    except (TypeError, ValueError):
        return None


def check_device_binding(device: str, script: str = "训练脚本") -> None:
    """防呆：torchrun 多进程下写了**固定序号**的设备时直接退出。

    这是 check_ddp_launch 之外的第二条同类陷阱，且更隐蔽：`--ddp` 加了、进程组建起来了、
    `[dist]` 与 `[loaders]` 两行日志都正常打出来，但 `resolve_device` 见到 ":" 就原样返回
    （那是为单卡指定卡号服务的），于是 8 个进程的 `model.to("cuda:0")` 全落到 0 号卡。
    报出来的仍然是一句与根因无关的 CUDA out of memory。

    只拦"序号 ≠ 本 rank 的 LOCAL_RANK"的情况：显式写 `cuda:3` 且自己就是 local_rank 3
    是自洽的（少见但合法，比如手工用 CUDA_VISIBLE_DEVICES 排列过），放行。
    单卡（无 WORLD_SIZE 或 =1）时本函数恒为空操作。
    """
    if not launched_distributed():
        return
    idx = _explicit_cuda_index(device)
    if idx is None or idx == env_local_rank():
        return
    sys.exit(
        f"[dist] 错误: 检测到 torchrun 多进程启动（WORLD_SIZE={env_world_size()}），"
        f"但 --device 写成了固定序号 {device!r}。\n"
        f"[dist]   本进程 LOCAL_RANK={env_local_rank()}，继续跑会让多个进程挤到 "
        f"cuda:{idx} 而 OOM，且报错信息与根因无关。\n"
        f"[dist]   修法: 给 {script} 传 --device cuda（不带序号），"
        f"由框架按 LOCAL_RANK 自动绑卡。")


def resolve_device(device: str) -> str:
    """把 "cuda" 解析成本 rank 该用的 "cuda:{LOCAL_RANK}"；已带序号或非 cuda 时原样返回。

    已带序号时**先过 check_device_binding**：多进程下写死 cuda:0 是 8 卡训练最隐蔽的
    一种失败（见该函数），在这里拦一次比让它 OOM 强得多。单卡时该检查是空操作。
    """
    if not device.startswith("cuda") or ":" in device:
        check_device_binding(device, "训练脚本")
        return device
    if not launched_distributed():
        return device
    return f"cuda:{env_local_rank()}"


def maybe_init_distributed(ddp: bool, backend: str = "nccl",
                           device: str = "cuda", verbose: bool = True) -> bool:
    """幂等地建立进程组，返回本进程是否处于分布式模式。

    顺序是硬性要求，不是风格问题：**`torch.cuda.set_device` 必须在
    `init_process_group` 之前**。不这么做的话每个 rank 的 current device 都是 0，
    NCCL 会把通信器建在 0 号卡上，表现为 8 卡启动即挂死或
    `duplicate GPU detected`——而且这类报错完全指不到根因。

    backend 会按实际可用性纠正：没有 CUDA 时 nccl 不可用，自动降级 gloo
    （让多进程 CPU 冒烟测试可跑）。
    """
    if not ddp:
        return False
    if process_group_ready():
        return True

    local = env_local_rank()
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local)          # ← 必须在 init_process_group 之前
    elif backend == "nccl":
        backend = "gloo"                      # 无 CUDA 时 nccl 不可用

    dist.init_process_group(backend=backend)
    if verbose and dist.get_rank() == 0:
        print(f"[dist] 进程组已建立: backend={backend}, world_size={dist.get_world_size()}"
              + (f", 本 rank 绑定 cuda:{local}" if use_cuda else ""))
    return True


def check_ddp_launch(ddp: bool, script: str = "训练脚本") -> None:
    """防呆：torchrun 起了多进程但没开 DDP 时直接退出，而不是让 8 个进程挤爆 0 号卡。

    这是最容易犯且最难诊断的一步：`train.ddp` 漏配时 Trainer 会走单卡分支、
    `cfg.device` 保持裸 "cuda"，8 个进程全部 `model.to("cuda")` 落到 GPU0，
    报出来的是一句与根因无关的 CUDA out of memory。
    """
    if launched_distributed() and not ddp:
        sys.exit(
            f"[dist] 错误: 检测到 torchrun 多进程启动（WORLD_SIZE={env_world_size()}），"
            f"但 DDP 未开启。\n"
            f"[dist]   继续跑会让 {env_world_size()} 个进程全部挤到 cuda:0 而 OOM，"
            f"且报错信息与根因无关。\n"
            f"[dist]   修法二选一：给 {script} 加 --ddp，"
            f"或在 YAML 的 train: 段写 ddp: true")


# --------------------------- 训练脚本的统一入口 ---------------------------
def add_ddp_args(ap) -> None:
    """给 argparse 挂上四个脚本共用的 DDP 开关。"""
    ap.add_argument("--ddp", action="store_true", default=None,
                    help="启用 DDP 多卡训练。torchrun 启动但没开会直接报错退出")
    ap.add_argument("--no-ddp", dest="ddp", action="store_false", help="强制关闭 DDP")
    ap.add_argument("--ddp-backend", default=None, help="nccl(GPU) | gloo(CPU)")


def resolve_ddp(cli_ddp: bool | None, cli_backend: str | None,
                yaml_train: dict | None, script: str = "训练脚本",
                device: str | None = None) -> tuple[bool, str]:
    """解析 DDP 开关（命令行 > YAML > 关）并立刻做启动方式防呆。返回 (ddp, backend)。

    必须在**加载模型 / 建 DataLoader 之前**调用：漏配 ddp 时 8 个进程会全部挤到
    cuda:0，等到那时才 OOM 的话，已经白白加载了 8 份底座权重，而报错信息指不到根因。

    这里**只判定不建组**。进程组要等 DataLoader worker fork 完（即 Trainer 构造时）
    再建，否则 `torch.cuda.set_device` 会在 fork 之前创建 CUDA 上下文，破坏训练脚本
    刻意安排的顺序。数据分片不依赖进程组——它直接读 torchrun 的环境变量，见 dist_info。

    device 给了就一并做设备绑定防呆（见 check_device_binding）。传它的价值和上面一样：
    在加载 4B 底座之前退出，而不是等 8 份权重都进了 0 号卡才 OOM。
    """
    yt = yaml_train or {}
    ddp = cli_ddp if cli_ddp is not None else bool(yt.get("ddp", False))
    backend = cli_backend or yt.get("ddp_backend") or "nccl"
    check_ddp_launch(ddp, script)
    if device is not None:
        check_device_binding(device, script)
    return ddp, backend
