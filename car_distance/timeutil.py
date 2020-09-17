import datetime


def get_epochtime_ms():
    return round(datetime.datetime.utcnow().timestamp() * 1000)