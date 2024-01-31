import glob
import logging
import requests

from pathlib import Path


class SecurionSource:

    def __init__(self, unity_code: str, png_cache_dir=None, server_url=None):
        self.png_cache_dir = png_cache_dir
        self.server_url = server_url

        self.valid_url_codes = None
        self.valid_file_codes = None

        unity_code = Path(unity_code)
        unity_code = unity_code.name.split('.')[0]
        unity_code = unity_code.replace("_(1)", "").replace("_(2)", "").replace("_(3)", "")

        self.unity_code = unity_code

        unity_code_len = len(self.unity_code)

        self._failed = False

        if unity_code_len == 67:
            self.code_type = 'video'
            self.unity_i_code = self.unity_code
        elif unity_code_len == 72:
            self.code_type = 'frame'
            self.frame_num = int(unity_code[-4:])
            self.unity_i_code = self.unity_code[:-5]
            self.unity_f_code = self.unity_code
        else:
            logging.error(f"{unity_code} not a valid code")
            self._failed = True
            raise Exception

        self._sub_a = self.unity_i_code[:2]
        self._sub_b = self.unity_i_code[3:5]
        self._sub_c = self.unity_i_code[5:7]

    def get_frame_path(self, frame_offset=0):
        if self._failed:
            return ""

        if self.code_type == 'frame':
            return f"{self.png_cache_dir / self._sub_a / self._sub_b / self._sub_c / self.unity_i_code}-{(self.frame_num + frame_offset):04}.png"
        else:
            raise Exception

    def get_frame_url(self, frame_offset=0, frame_num=None):
        if self._failed:
            return ""

        if frame_num is not None:
            location = f"{self.server_url}/{self._sub_a}/{self._sub_b}/{self._sub_c}/{self.unity_i_code}-{frame_num:04}.png"
        elif self.code_type == 'frame':
            location = f"{self.server_url}/{self._sub_a}/{self._sub_b}/{self._sub_c}/{self.unity_i_code}-{(self.frame_num + frame_offset):04}.png"
        else:
            raise Exception

        return location

    def get_all_frames_path(self):
        search_string = f"{self.png_cache_dir / self._sub_a / self._sub_b / self._sub_c / self.unity_i_code}*.png"
        images_path = glob.glob(search_string)
        valid_codes = sorted(images_path)
        logging.info(f"{self.unity_i_code} Number of frames {len(valid_codes)}")

        return valid_codes

    def get_all_frames_url(self):
        i = 0
        valid_codes = []
        valid_urls = []
        while True:
            source = self.get_frame_url(frame_num=i)
            r = requests.head(source)
            if r.status_code == 200:
                valid_urls.append(source)
                valid_codes.append(f"{self.unity_i_code}-{i:04}")
                i = i + 1
            else:
                break

        num_valid_urls = len(valid_codes)
        if num_valid_urls == 0:
            logging.error(f"{self.unity_code} has {num_valid_urls} valid frame urls")
        else:
            logging.info(f"{self.unity_code} has {num_valid_urls} valid frame urls")

        self.valid_url_codes = valid_codes
        return valid_urls
        #return

    def get_valid_url_codes(self):
        if not self.valid_url_codes:
            self.get_all_frames_url()

        return self.valid_url_codes