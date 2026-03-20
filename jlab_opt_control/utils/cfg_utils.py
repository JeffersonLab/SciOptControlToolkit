import logging

cfg_log = logging.getLogger("CfgLogger")
cfg_log.setLevel(logging.INFO)


def cfg_get(obj, key, default=None):
    try:
        obj = obj[key]
    except KeyError:
        cfg_log.error(f'Problem with key request using {default}')
        return default
    return obj
