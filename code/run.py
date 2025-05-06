# -*- coding: utf-8 -*-
import logging
from fastxc import FastXCPipeline, StepMode

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config_path = "./config/test_c3.ini"
    pipeline = FastXCPipeline(config_path)

    # 比如想让:
    #  - GenerateFilter: 全部执行(默认)
    #  - OrganizeSAC: 只生成必要文件，不执行(其实它也没有"执行"概念，这里只是演示)
    #  - Sac2Spec: 跳过
    #  - XC: 只生成命令，不部署
    #  - Stack: 全部执行
    #  - Rotate: 跳过

    # 各个步骤的执行模式
    # StepMode.ALL: 全部执行
    # StepMode.PREPARE_ONLY: 只生成必要文件，不执行
    # StepMode.SKIP: 跳过
    # StepMode.CMD_ONLY: 只生成命令，不部署
    steps_config = {
        "GenerateFilter": StepMode.ALL,
        "OrganizeSAC": StepMode.ALL,
        "Sac2Spec": StepMode.ALL,
        "CrossCorrelation": StepMode.ALL,
        "ConcatenateNcf": StepMode.ALL,
        "Stack": StepMode.ALL,
        "Rotate": StepMode.ALL,
        "Sac2Dat": StepMode.ALL,
    }

    pipeline.run(steps_config)
