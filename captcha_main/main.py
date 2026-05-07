# -*- coding: utf-8 -*-
import base64
import logging
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import cv2
from curl_cffi import requests


class WaimaoxiaCaptchaClient:
    """
    外贸侠旋转验证码处理客户端

    Note:
        使用 curl_cffi 的 requests.Session 维持原始请求行为，
        保持现有请求参数、请求头与验证码处理流程不变。
    """

    CAPTCHA_URL = "https://captcha.waimaoxia.net/api/rotate/basic/captcha"
    VERIFY_URL = "https://captcha.waimaoxia.net/api/rotate/basic/verify"
    CAPTCHA_IMAGE_PATH = Path("captcha.png")
    THUMB_IMAGE_PATH = Path("captcha_2.png")

    def __init__(self, mask_log_url: bool = False) -> None:
        """
        初始化验证码客户端

        Note:
            默认保留直连占位，便于后续切换代理而不改动主流程。
        """
        self.log = self._build_logger()
        self.session = requests.Session(impersonate="chrome146")
        self.proxies = None
        self.mask_log_url = mask_log_url

    def _build_logger(self) -> logging.Logger:
        """
        构建日志记录器

        Returns:
            logging.Logger: 当前类专用日志对象
        """
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def format_log_url(self, url: str) -> str:
        """
        格式化日志中的 URL

        Args:
            url (str): 原始请求 URL

        Returns:
            str: 按配置返回原始或打码后的 URL

        Note:
            仅影响日志输出，不改动真实请求参数。
        """
        if not self.mask_log_url:
            return url

        split_result = urlsplit(url)
        path_parts = [part for part in split_result.path.split("/") if part]

        if not path_parts:
            masked_path = "/***"
        elif len(path_parts) == 1:
            masked_path = f"/***{path_parts[-1]}"
        else:
            masked_path = "/" + "/".join([path_parts[0], "***", path_parts[-1]])

        masked_query = "&".join(
            f"{key}=***"
            for key, _ in parse_qsl(split_result.query, keep_blank_values=True)
        )

        return urlunsplit(
            (
                split_result.scheme,
                split_result.netloc,
                masked_path,
                masked_query,
                split_result.fragment,
            )
        )

    @staticmethod
    def build_captcha_headers() -> Dict[str, str]:
        """
        构建获取验证码请求头

        Returns:
            dict: 获取验证码接口所需请求头
        """
        return {
            "accept": "application/json, text/plain, */*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cache-control": "no-cache",
            "origin": "https://www.waimaoxia.net",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://www.waimaoxia.net/",
            "sec-ch-ua": "\"Google Chrome\";v=\"147\", \"Not.A/Brand\";v=\"8\", \"Chromium\";v=\"147\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
        }

    @staticmethod
    def build_verify_headers() -> Dict[str, str]:
        """
        构建验证码校验请求头

        Returns:
            dict: 验证码校验接口所需请求头
        """
        return {
            "accept": "application/json, text/plain, */*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cache-control": "no-cache",
            "content-type": "application/x-www-form-urlencoded",
            "origin": "https://www.waimaoxia.net",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://www.waimaoxia.net/",
            "sec-ch-ua": "\"Google Chrome\";v=\"147\", \"Not.A/Brand\";v=\"8\", \"Chromium\";v=\"147\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
        }

    @staticmethod
    def decode_base64_image(image_base64: str) -> bytes:
        """
        解码 Base64 图片数据

        Args:
            image_base64 (str): 带 data:image 前缀的 Base64 字符串

        Returns:
            bytes: 解码后的图片二进制内容
        """
        base64_data = image_base64.split(",", maxsplit=1)[1]
        return base64.b64decode(base64_data)

    def save_image(self, image_bytes: bytes, file_path: Path) -> None:
        """
        保存图片到本地

        Args:
            image_bytes (bytes): 图片字节流
            file_path (Path): 目标文件路径
        """
        self.log.info(f"[存储模块] 开始保存图片 - 路径: {file_path}")
        with file_path.open("wb") as file:
            file.write(image_bytes)
        self.log.info(f"[存储模块] 图片保存完成 - 路径: {file_path}")

    @staticmethod
    def calculate_rotation_angle(bg_path: str, small_path: str) -> int:
        """
        识别小图相对大图需要旋转的角度

        Args:
            bg_path (str): 背景图路径
            small_path (str): 小图路径

        Returns:
            int: 最终提交到验证码接口的旋转角度

        Raises:
            ValueError: 图片读取失败时抛出异常
        """
        bg = cv2.imread(bg_path)
        small = cv2.imread(small_path)
        if bg is None or small is None:
            raise ValueError(f"图片读取失败，bg_path={bg_path}, small_path={small_path}")

        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        best_angle = 0
        best_score = -1.0

        height, width = small_gray.shape
        crop = small_gray[20:height - 20, 20:width - 20]

        for angle in range(360):
            rotate_matrix = cv2.getRotationMatrix2D(
                (crop.shape[1] // 2, crop.shape[0] // 2),
                angle,
                1.0,
            )
            rotated = cv2.warpAffine(
                crop,
                rotate_matrix,
                (crop.shape[1], crop.shape[0]),
            )
            result = cv2.matchTemplate(
                bg_gray,
                rotated,
                cv2.TM_CCOEFF_NORMED,
            )
            score = result.max()
            if score > best_score:
                best_score = score
                best_angle = angle

        return (360 - best_angle) % 360

    def fetch_captcha(self) -> Tuple[str, str, str]:
        """
        请求验证码接口并提取关键字段

        Returns:
            tuple: (captcha_key, image_base64, thumb_base64)

        Raises:
            KeyError: 接口返回缺少关键字段时抛出异常

        Note:
            保持原始 curl_cffi 指纹与请求头不变。
        """
        headers = self.build_captcha_headers()
        self.log.info(
            f"[请求模块] 开始请求验证码 - URL: {self.format_log_url(self.CAPTCHA_URL)}"
        )
        self.log.info(
            f"[请求模块] 当前代理：{'直连' if not self.proxies else self.proxies}"
        )
        response = self.session.get(self.CAPTCHA_URL, headers=headers)
        self.log.info(
            f"[请求模块] 验证码请求完成 - 状态码: {response.status_code}"
        )

        captcha = response.json()
        captcha_key = captcha["captcha_key"]
        image_base64 = captcha["image_base64"]
        thumb_base64 = captcha["thumb_base64"]
        return captcha_key, image_base64, thumb_base64

    def verify_captcha(self, captcha_key: str, angle: int) -> str:
        """
        提交旋转角度并获取校验结果

        Args:
            captcha_key (str): 验证码会话键
            angle (int): 识别出的旋转角度

        Returns:
            str: 验证接口原始响应文本

        Note:
            保持原始表单字段与请求头不变。
        """
        headers = self.build_verify_headers()
        data = {
            "angle": str(angle),
            "key": captcha_key,
        }
        self.log.info(
            f"[请求模块] 开始提交验证码校验 - URL: {self.format_log_url(self.VERIFY_URL)} | angle: {angle}"
        )
        response = self.session.post(self.VERIFY_URL, headers=headers, data=data)
        self.log.info(
            f"[请求模块] 验证码校验完成 - 状态码: {response.status_code}"
        )
        return response.text

    def get_login_token(self) -> str:
        """
        获取验证码校验结果

        Returns:
            str: 验证接口返回文本

        Raises:
            Exception: 任一环节失败时记录日志后继续抛出

        Note:
            保持下载验证码、落盘、识别角度、提交校验的原始逻辑不变。
        """
        try:
            captcha_key, image_base64, thumb_base64 = self.fetch_captcha()

            image_bytes = self.decode_base64_image(image_base64)
            thumb_bytes = self.decode_base64_image(thumb_base64)

            self.save_image(image_bytes, self.CAPTCHA_IMAGE_PATH)
            self.save_image(thumb_bytes, self.THUMB_IMAGE_PATH)

            angle = self.calculate_rotation_angle(
                str(self.CAPTCHA_IMAGE_PATH),
                str(self.THUMB_IMAGE_PATH),
            )
            self.log.info(f"[解析模块] 角度识别完成 - angle: {angle}")

            verify_result = self.verify_captcha(captcha_key, angle)
            self.log.info("[主流程] 验证码处理完成")
            return verify_result
        except (KeyError, ValueError, IndexError, requests.RequestsError) as exc:
            self.log.error(f"{self.__class__.__name__} > get_login_token > {exc}")
            raise


def main() -> None:
    """
    脚本主入口

    Note:
        保持直接执行方式不变，不引入命令行参数。
    """
    client = WaimaoxiaCaptchaClient(mask_log_url=True)
    print(client.get_login_token())


if __name__ == "__main__":
    main()
