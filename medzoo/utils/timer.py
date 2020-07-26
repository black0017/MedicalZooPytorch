import time


class Timer:
    DEFAULT_TIME_FORMAT_DATE_TIME = "%Y/%m/%d-%H:%M:%S"
    DEFAULT_TIME_FORMAT = ["%03dms", "%02ds", "%02dm", "%02dh"]

    def __init__(self):
        self.start = time.time() * 1000

    def get_current(self):
        return self.get_time(self.start)

    def reset(self):
        self.start = time.time() * 1000

    def get_time_since_start(self, time_format=None):
        return self.get_time(self.start, time_format)

    def get_time(self, start=None, end=None, time_format=None):

        if start is None:
            if time_format is None:
                time_format = self.DEFAULT_TIME_FORMAT_DATE_TIME

            return time.strftime(time_format)

        if end is None:
            end = time.time() * 1000
        time_elapsed = end - start

        if time_format is None:
            time_format = self.DEFAULT_TIME_FORMAT

        s, ms = divmod(time_elapsed, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)

        items = [ms, s, m, h]
        assert len(items) == len(time_format), "Format length should be same as items"

        time_str = ""
        for idx, item in enumerate(items):
            if item != 0:
                time_str = time_format[idx] % item + " " + time_str

        # Means no more time is left
        if len(time_str) == 0:
            time_str = "0ms"

        return time_str.strip()

