#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from fastxc import FastXCPipeline


def main():
    """
    生成模板配置文件。
    如果有传入命令行参数，就以该参数作为目标文件名，否则默认写到 'template_config.ini.copy'。
    """
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "./config/template_config.ini"

    print(f"Generating template config file to: {output_path}")
    FastXCPipeline.generate_template_config(output_path)


if __name__ == "__main__":
    main()
