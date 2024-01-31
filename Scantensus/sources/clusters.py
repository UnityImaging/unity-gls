import logging
logger = logging.getLogger()


class ClusterSource:

    def __init__(self, unity_code: str, png_cache_dir=None, server_url=None):
        self.png_cache_dir = png_cache_dir
        self.server_url = server_url

        self.unity_code = unity_code
        self.unity_f_code = unity_code

        self._sub1 = unity_code.split("~")[0]

        self._failed = False
        self.code_type = 'frame'

    def get_frame_path(self, frame_offset=0):
        if self._failed:
            return ""

        if self.code_type == 'frame':
            return f"{self.png_cache_dir / self._sub1 / self.unity_code}"
        else:
            raise Exception

    def get_frame_url(self, frame_offset=0):
        if self._failed:
            return ""

        if self.code_type == 'frame':
            return f"{self.server_url}/{self.unity_code}"
        else:
            raise Exception

    def get_all_frames_path(self):
        raise Exception(f"Source was Clusters - you only get a single frame")

    def get_all_frames_url(self):
        raise Exception(f"Source was Clusters - you only get a single frame")
