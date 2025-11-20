# 安装环境

## 安装要求

-   **Python 版本**：{bdg-danger-line}`3.10.*`

    | Python 版本 | 支持状态 |
    | :---------: | :------: |
    |    ≤ 3.9    |    ❌    |
    |    3.10     |    ✅    |
    |   ≥ 3.11    |    ❌    |

-   **包管理器**：{bdg-danger-line}`UV`
    [UV 安装参考](https://docs.astral.sh/uv/getting-started/installation/)

-   **系统及架构**：

    -   {bdg-danger-line}`Windows(x86_64)`
    -   {bdg-danger-line}`Linux(x86_64)`

    ```{note}
    各平台支持的功能如下：

    | 操作系统 | CPU 仿真 | 交互式查看器 | GPU 仿真 |
    | :------: | :------: | :----------: | :------: |
    |  Linux   |    ✅    |      ✅      |    🛠️ 开发中    |
    | Windows  |    ✅    |      ✅      |    🛠️ 开发中    |
    ```

## 安装方法

### 克隆项目

```bash
git clone https://github.com/Motphys/MotrixLab.git
cd MotrixLab
```

### 安装依赖

使用 UV 安装项目依赖：

```bash
# 安装所有依赖
uv sync --all-packages --all-extras
```

如果只需要安装一种训练后端，可以选择单独安装指定的后端类型：

```bash

# 安装 SKRL JAX （仅支持 Linux 平台）
uv sync --all-packages --extra skrl-jax

# 安装 SKRL PyTorch
uv sync --all-packages --extra skrl-torch
```
