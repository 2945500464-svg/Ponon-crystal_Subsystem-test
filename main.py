
"""
不同工况下激振器对副车架激励下传递损失试验数据处理。

主入口保持为：python main.py
实际功能已拆分到 chengxu 包，便于人工检查和维护。
"""

from __future__ import annotations

from chengxu.config import *  # noqa: F401,F403
from chengxu.utils import *  # noqa: F401,F403
from chengxu.mat_io import *  # noqa: F401,F403
from chengxu.signal_processing import *  # noqa: F401,F403
from chengxu.data_format import *  # noqa: F401,F403
from chengxu.analysis import *  # noqa: F401,F403
from chengxu.plotting import *  # noqa: F401,F403
from chengxu.plot_style import *  # noqa: F401,F403
from chengxu.exporters import *  # noqa: F401,F403
from chengxu.gui import launch_analysis_gui


def main() -> None:
    launch_analysis_gui()


if __name__ == "__main__":
    main()
