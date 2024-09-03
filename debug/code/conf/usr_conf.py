#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :usr_conf.py
@Author  :kaiwu
@Date    :2023/7/1 10:37

"""


def usr_conf_check(usr_conf, logger):
    if not usr_conf:
        logger.error("usr_conf is None, please check")
        return False

    diy = usr_conf.get("diy", {})
    start = diy.get("start")
    end = diy.get("end")
    treasure_id = diy.get("treasure_id")
    talent_type = diy.get("talent_type")
    treasure_num = diy.get("treasure_num")
    treasure_random = diy.get("treasure_random")
    max_step = diy.get("max_step")

    if treasure_id and len(treasure_id) != len(set(treasure_id)):
        logger.error("Duplicate elements in treasure_id, please check")
        return False

    if treasure_id and not set(treasure_id).issubset(set(range(1, 16))):
        logger.error("Elements in treasure_id should be between 1 and 15, please check")
        return False

    if treasure_num and not (0 <= treasure_num <= 13):
        logger.error("treasure_num should be between 0 and 13, please check")
        return False

    if treasure_random is not None and treasure_random not in [0, 1]:
        logger.error("treasure_random field can only be 0 or 1, please check")
        return False

    if start is not None and end is not None and start == end:
        logger.error("Start and end points should not be the same, please check")
        return False

    if start is not None and (start < 1 or start > 15):
        logger.error("Start should be between 1 and 15, please check")
        return False

    if end is not None and (end < 1 or end > 15):
        logger.error("End should be between 1 and 15, please check")
        return False

    if talent_type is not None and talent_type != 1:
        logger.error("talent_type field can only be 1, please check")
        return False

    if max_step is not None:
        try:
            max_step = int(max_step)
            if max_step < 1:
                logger.error("max_step should not be negative or 0, please check")
                return False
        except ValueError:
            logger.error("max_step should be an integer, please check")
            return False

    if not treasure_random:
        if not set(treasure_id).issubset(set([i for i in range(1, 16)])):
            logger.error("treasure_random field can only be 0 or 1, please check")
            return False

        if start in treasure_id or end in treasure_id:
            logger.error(f"treasure_id should not include the start or end points, set to {treasure_id}, start {start}, end {end}")
            return False

    if treasure_random:
        if treasure_num not in [i for i in range(0, 14)]:
            logger.error(f"treasure_num should be between 0 and 13, set to {treasure_num}")
            return False

    return True
