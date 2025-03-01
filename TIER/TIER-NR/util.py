# -*- coding: utf-8 -*-


import torch
import logging
import math
import torch.nn as nn


def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)


