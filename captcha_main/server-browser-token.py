# -*- coding: utf-8 -*-
"""固定浏览器上下文的 Semrush Cookie 服务。"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from playwright.async_api import (
    BrowserContext,
    Error as PlaywrightError,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)
from captcha_main.config import SEMRUSH_PASSWORD, SEMRUSH_USERNAME


def build_success_response(data: dict[str, Any]) -> dict[str, Any]:
    """
    构建成功响应

    Args:
        data (dict[str, Any]): 返回数据

    Returns:
        dict[str, Any]: 统一成功响应
    """

    return {"code": 200, "data": data}


def build_fail_response(message: str) -> dict[str, Any]:
    """
    构建失败响应

    Args:
        message (str): 失败原因

    Returns:
        dict[str, Any]: 统一失败响应
    """

    return {"code": 500, "data": [], "message": message}


class SemrushBrowserTokenService:
    """
    Semrush 固定浏览器 Cookie 服务。

    Note:
        使用 Playwright persistent context 维持稳定资料目录，
        通过浏览器自动化登录、点击目标入口并抓取指定 Cookie。
    """

    def __init__(self) -> None:
        """
        初始化固定浏览器服务配置

        Returns:
            None
        """

        self.log = self._build_logger()
        self.base_dir = Path(__file__).resolve().parent
        self.user_data_dir = self.base_dir / "playwright-profile"
        self.local_sign_url = "http://127.0.0.1:8080/index.html"
        self.origin_url = "https://www.waimaoxia.net"
        self.login_page_url = f"{self.origin_url}/login"
        self.dashboard_url = "https://www.waimaoxia.net/dashboard"
        self.login_url = "https://www.waimaoxia.net/api/user/login"
        self.captcha_url = "https://captcha.waimaoxia.net/api/rotate/basic/captcha"
        self.verify_url = "https://captcha.waimaoxia.net/api/rotate/basic/verify"
        self.semrush_entry_text = "Semrush（商业版）"
        self.timeout = 30
        self.retry = 3
        self.headless = False
        self.browser_proxy = os.getenv("SEMRUSH_BROWSER_PROXY", "").strip()
        self.username = str(SEMRUSH_USERNAME).strip()
        self.password = str(SEMRUSH_PASSWORD).strip()
        self.username_selector = "input[placeholder='请输入登录账号']"
        self.password_selector = "input[placeholder='请输入登录密码']"
        self.agree_checkbox_selector = "input[type='checkbox']"
        self.login_button_selector = "div.login-btn"
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/147.0.0.0 Safari/537.36"
        )
        self.context_options = {
            "user_agent": self.user_agent,
            "locale": "zh-CN",
            "timezone_id": "Asia/Shanghai",
            "viewport": {"width": 1280, "height": 720},
        }
        self.launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--start-maximized",
            "--window-size=1280,720",
        ]
        self.playwright: Playwright | None = None
        self.context: BrowserContext | None = None
        self.dashboard_page: Page | None = None
        self.sign_page: Page | None = None
        self.browser_lock = asyncio.Lock()

    def _build_logger(self) -> logging.Logger:
        """
        构建日志对象

        Returns:
            logging.Logger: 日志实例
        """

        logger = logging.getLogger(self.__class__.__name__)
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    def calculate_rotation_candidates(
        self,
        bg_source: str,
        small_source: str,
        limit: int = 5,
    ) -> list[int]:
        """
        计算旋转验证码候选角度列表

        Args:
            bg_source (str): 背景图 data URL
            small_source (str): 小图 data URL
            limit (int): 返回候选角度数量

        Returns:
            list[int]: 候选旋转角度列表

        Raises:
            RuntimeError: 图片解码失败时抛出异常
        """

        if "," not in bg_source or "," not in small_source:
            raise RuntimeError("验证码图片数据格式异常")

        bg_buffer = np.frombuffer(
            base64.b64decode(bg_source.split(",", 1)[1]),
            dtype=np.uint8,
        )
        small_buffer = np.frombuffer(
            base64.b64decode(small_source.split(",", 1)[1]),
            dtype=np.uint8,
        )
        bg = cv2.imdecode(bg_buffer, cv2.IMREAD_COLOR)
        small = cv2.imdecode(small_buffer, cv2.IMREAD_COLOR)
        if bg is None or small is None:
            raise RuntimeError("验证码图片读取失败")

        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        height, width = small_gray.shape
        crop = small_gray[20 : height - 20, 20 : width - 20]

        scored_angles: list[tuple[float, int]] = []
        for angle in range(360):
            matrix = cv2.getRotationMatrix2D(
                (crop.shape[1] // 2, crop.shape[0] // 2),
                angle,
                1.0,
            )
            rotated = cv2.warpAffine(crop, matrix, (crop.shape[1], crop.shape[0]))
            result = cv2.matchTemplate(bg_gray, rotated, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            scored_angles.append((score, (360 - angle) % 360))

        scored_angles.sort(key=lambda item: item[0], reverse=True)
        candidate_angles: list[int] = []
        for _, candidate in scored_angles:
            if candidate not in candidate_angles:
                candidate_angles.append(candidate)
            if len(candidate_angles) >= limit:
                break

        return candidate_angles

    async def wait_login_dependencies(self, page: Page) -> None:
        """
        等待登录页依赖加载完成

        Args:
            page (Page): Playwright 页面对象

        Returns:
            None
        """

        await page.wait_for_selector(self.username_selector, timeout=self.timeout * 1000)
        await page.wait_for_selector(self.password_selector, timeout=self.timeout * 1000)
        await page.wait_for_function(
            """
            () => {
                return (
                    typeof window.sign === "function" &&
                    !!localStorage.getItem("finger")
                );
            }
            """,
            timeout=self.timeout * 1000,
        )
        self.log.info("[请求模块] 登录页依赖加载完成")

    async def fetch_rotate_captcha(self, page: Page) -> dict[str, Any]:
        """
        通过浏览器上下文获取旋转验证码

        Args:
            page (Page): Playwright 页面对象

        Returns:
            dict[str, Any]: 统一响应，包含验证码图片与 key
        """

        self.log.info("[请求模块] 开始通过浏览器上下文获取旋转验证码")
        captcha_response = await page.evaluate(
            """
            async (captchaUrl) => {
                const response = await fetch(captchaUrl, {
                    method: "GET",
                    credentials: "include"
                });
                const data = await response.json();
                return { ok: response.ok, status: response.status, data };
            }
            """,
            self.captcha_url,
        )
        response_data = captcha_response.get("data", {})
        if not captcha_response.get("ok"):
            message = response_data.get("msg") or f"验证码接口状态异常: {captcha_response.get('status')}"
            self.log.error("[请求模块] 旋转验证码获取失败 - %s", message)
            return build_fail_response(message)

        captcha_key = str(response_data.get("captcha_key", ""))
        image_base64 = str(response_data.get("image_base64", ""))
        thumb_base64 = str(response_data.get("thumb_base64", ""))
        if response_data.get("code") not in (0, None) or not captcha_key or not image_base64 or not thumb_base64:
            message = response_data.get("msg") or "旋转验证码响应缺少关键字段"
            self.log.error("[请求模块] 旋转验证码数据异常 - %s", message)
            return build_fail_response(message)

        self.log.info("[请求模块] 旋转验证码获取成功 - key: %s", captcha_key)
        return build_success_response(
            {
                "captcha_key": captcha_key,
                "image_base64": image_base64,
                "thumb_base64": thumb_base64,
            }
        )

    async def verify_rotate_captcha(
        self,
        page: Page,
        angle: int,
        captcha_key: str,
    ) -> dict[str, Any]:
        """
        通过浏览器上下文校验旋转验证码
        Args:
            page (Page): Playwright 页面对象
            angle (int): 识别角度
            captcha_key (str): 验证码 key

        Returns:
            dict[str, Any]: 统一响应
        """

        self.log.info("[请求模块] 开始校验旋转验证码 - key: %s | angle: %s", captcha_key, angle)
        verify_response = await page.evaluate(
            """
            async ({verifyUrl, angle, captchaKey}) => {
                const body = new URLSearchParams({
                    angle: String(angle),
                    key: captchaKey
                });
                const response = await fetch(verifyUrl, {
                    method: "POST",
                    headers: {
                        "content-type": "application/x-www-form-urlencoded; charset=UTF-8"
                    },
                    body: body.toString(),
                    credentials: "include"
                });
                const data = await response.json();
                if ((data.code ?? 0) === 0) {
                    window.captcha_key = captchaKey;
                } else {
                    window.captcha_key = "";
                }
                return { ok: response.ok, status: response.status, data };
            }
            """,
            {
                "verifyUrl": self.verify_url,
                "angle": angle,
                "captchaKey": captcha_key,
            },
        )
        response_data = verify_response.get("data", {})
        if not verify_response.get("ok"):
            message = response_data.get("msg") or f"验证码校验接口状态异常: {verify_response.get('status')}"
            self.log.error("[请求模块] 旋转验证码校验失败 - %s", message)
            return build_fail_response(message)
        if response_data.get("code") not in (0, None):
            message = response_data.get("msg") or "旋转验证码校验失败"
            self.log.error("[请求模块] 旋转验证码业务校验失败 - %s", message)
            return build_fail_response(message)

        self.log.info("[请求模块] 旋转验证码校验成功 - key: %s", captcha_key)
        return build_success_response({"captcha_key": captcha_key, "angle": angle})

    async def solve_rotate_captcha(self, page: Page) -> dict[str, Any]:
        """
        自动识别并校验旋转验证码

        Args:
            page (Page): Playwright 页面对象

        Returns:
            dict[str, Any]: 统一响应

        Note:
            当验证码校验失败时，自动重新拉取新验证码并重试。
        """

        last_message = "旋转验证码校验失败"
        for captcha_attempt in range(1, self.retry + 1):
            captcha_response = await self.fetch_rotate_captcha(page)
            if captcha_response.get("code") != 200:
                last_message = captcha_response.get("message", "旋转验证码获取失败")
                self.log.error(
                    "[会话模块] 验证码获取失败 - 重试: %s/%s | 错误: %s",
                    captcha_attempt,
                    self.retry,
                    last_message,
                )
                if captcha_attempt < self.retry:
                    await asyncio.sleep(2 ** captcha_attempt)
                continue

            captcha_data = captcha_response.get("data", {})
            captcha_key = str(captcha_data.get("captcha_key", ""))
            image_base64 = str(captcha_data.get("image_base64", ""))
            thumb_base64 = str(captcha_data.get("thumb_base64", ""))
            angle_candidates = self.calculate_rotation_candidates(
                image_base64,
                thumb_base64,
                limit=5,
            )
            self.log.info(
                "[请求模块] 旋转验证码候选角度识别完成 - candidates: %s",
                angle_candidates,
            )

            for angle in angle_candidates:
                verify_response = await self.verify_rotate_captcha(page, angle, captcha_key)
                if verify_response.get("code") == 200:
                    return verify_response
                last_message = verify_response.get("message", "旋转验证码校验失败")

            self.log.error(
                "[会话模块] 验证码校验失败，自动重试 - 重试: %s/%s | 错误: %s",
                captcha_attempt,
                self.retry,
                last_message,
            )
            if captcha_attempt < self.retry:
                await asyncio.sleep(2 ** captcha_attempt)

        return build_fail_response(last_message)

    async def get_page_login_state(self, page: Page) -> dict[str, Any]:
        """
        获取页面登录状态快照

        Args:
            page (Page): Playwright 页面对象

        Returns:
            dict[str, Any]: 当前页面登录态信息
        """

        state = await page.evaluate(
            """
            () => {
                return {
                    href: location.href,
                    pathname: location.pathname,
                    has_token_cookie: document.cookie.includes("token="),
                    local_token: localStorage.getItem("token") || "",
                    page_text: (document.body?.innerText || "").slice(0, 5000)
                };
            }
            """
        )
        state["is_logged_in"] = bool(
            state.get("has_token_cookie")
            or state.get("pathname", "").startswith("/dashboard")
            or self.semrush_entry_text in state.get("page_text", "")
        )
        return state

    def build_cookies(self, token: str) -> list[dict[str, Any]]:
        """
        构建最小登录 cookie 集合

        Args:
            token (str): 登录 token

        Returns:
            list[dict[str, Any]]: Playwright 格式 cookie
        """

        return [
            {
                "name": "token",
                "value": token,
                "domain": ".waimaoxia.net",
                "path": "/",
            }
        ]

    async def start(self) -> None:
        """
        启动固定浏览器上下文

        Returns:
            None
        """

        async with self.browser_lock:
            if self.context is not None:
                self.log.info("[主流程] 固定浏览器已启动，无需重复初始化")
                return

            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            self.log.info("[主流程] 启动固定浏览器 - profile: %s", self.user_data_dir)
            self.playwright = await async_playwright().start()
            launch_options: dict[str, Any] = {
                "user_data_dir": str(self.user_data_dir),
                "headless": self.headless,
                "args": self.launch_args,
                **self.context_options,
            }
            if self.browser_proxy:
                launch_options["proxy"] = {"server": self.browser_proxy}
                self.log.info("[代理模块] 浏览器代理已启用 - %s", self.browser_proxy)
            else:
                self.log.info("[代理模块] 浏览器当前使用直连模式")

            self.context = await self.playwright.chromium.launch_persistent_context(
                **launch_options,
            )
            await self.context.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                """
            )
            await self.ensure_dashboard_page(force_reload=True)

    async def ensure_dashboard_page(self, force_reload: bool = False) -> Page:
        """
        确保 dashboard 页面可用

        Args:
            force_reload (bool): 是否强制重载

        Returns:
            Page: dashboard 页面对象
        """

        if self.context is None:
            raise RuntimeError("固定浏览器尚未启动")

        if self.dashboard_page is None or self.dashboard_page.is_closed():
            self.dashboard_page = (
                self.context.pages[0]
                if self.context.pages
                else await self.context.new_page()
            )
            self.log.info("[请求模块] 已创建 dashboard 页面")

        if force_reload or not self.dashboard_page.url.startswith(self.origin_url):
            self.log.info("[请求模块] 打开 dashboard - URL: %s", self.dashboard_url)
            await self.dashboard_page.goto(
                self.dashboard_url,
                wait_until="domcontentloaded",
                timeout=self.timeout * 1000,
            )

        return self.dashboard_page

    async def ensure_sign_page(self) -> Page:
        """
        确保本地 sign 页面可用

        Returns:
            Page: sign 页面对象
        """

        if self.context is None:
            raise RuntimeError("固定浏览器尚未启动")

        if self.sign_page is None or self.sign_page.is_closed():
            self.sign_page = await self.context.new_page()
            self.log.info("[请求模块] 已创建 sign 页面")

        self.log.info("[请求模块] 加载本地 sign 页 - URL: %s", self.local_sign_url)
        await self.sign_page.goto(
            self.local_sign_url,
            wait_until="domcontentloaded",
            timeout=self.timeout * 1000,
        )
        await self.sign_page.wait_for_function(
            "typeof window.sign === 'function'",
            timeout=self.timeout * 1000,
        )
        return self.sign_page

    async def get_sign(
        self,
        username: str,
        finger: str,
        captcha_key: str,
        url: str | None = None,
    ) -> dict[str, Any]:
        """
        通过固定浏览器上下文计算 sign

        Args:
            username (str): 登录用户名
            finger (str): 指纹 token
            captcha_key (str): 验证码 key
            url (str | None): 本地 sign 页地址

        Returns:
            dict[str, Any]: 统一响应，包含 sign
        """

        try:
            await self.start()
            async with self.browser_lock:
                if url:
                    self.local_sign_url = url
                page = await self.ensure_sign_page()
                result = await page.evaluate(
                    """
                    ([username, finger, captcha_key]) => {
                        return sign(
                            username,
                            finger,
                            captcha_key,
                            "",
                            "",
                            ""
                        );
                    }
                    """,
                    [username, finger, captcha_key],
                )
                self.log.info("[请求模块] sign 计算成功 - 用户: %s", username)
                return build_success_response({"sign": result})
        except (PlaywrightTimeoutError, PlaywrightError, RuntimeError) as exc:
            self.log.error("[请求模块] sign 计算失败 - 错误: %s", exc)
            return build_fail_response(str(exc))

    async def ensure_browser_login(self) -> dict[str, Any]:
        """
        通过浏览器自动化确保登录态有效

        Returns:
            dict[str, Any]: 统一响应

        Note:
            验证码失败时会自动重试，登录成功后复用当前浏览器上下文。
        """

        await self.start()
        if self.context is None:
            return build_fail_response("固定浏览器尚未启动")

        async with self.browser_lock:
            page = await self.ensure_dashboard_page(force_reload=True)
            login_state = await self.get_page_login_state(page)
            if login_state.get("is_logged_in"):
                self.log.info("[会话模块] 当前浏览器上下文已存在有效登录态")
                return build_success_response(
                    {
                        "mode": "browser",
                        "current_url": page.url,
                    }
                )

            if not self.username or not self.password:
                self.log.error("[会话模块] 缺少浏览器自动化登录凭证")
                return build_fail_response(
                    "缺少环境变量 SEMRUSH_USERNAME 或 SEMRUSH_PASSWORD"
                )

            for attempt in range(1, self.retry + 1):
                try:
                    self.log.info("[模拟操作] 准备执行浏览器自动化登录 - 第 %s 次", attempt)
                    await page.goto(
                        self.login_page_url,
                        wait_until="domcontentloaded",
                        timeout=self.timeout * 1000,
                    )
                    await self.wait_login_dependencies(page)

                    await page.locator(self.username_selector).fill(self.username)
                    self.log.info("[模拟操作] 登录账号已填充")
                    await page.locator(self.password_selector).fill(self.password)
                    self.log.info("[模拟操作] 登录密码已填充")
                    await page.locator(self.agree_checkbox_selector).check(force=True)
                    self.log.info("[模拟操作] 服务协议勾选完成")

                    verify_response = await self.solve_rotate_captcha(page)
                    if verify_response.get("code") != 200:
                        raise RuntimeError(
                            verify_response.get("message", "旋转验证码校验失败")
                        )

                    self.log.info("[模拟操作] 开始提交页面登录表单")
                    async with page.expect_response(
                        lambda response: (
                            self.login_url in response.url
                            and response.request.method.upper() == "POST"
                        ),
                        timeout=self.timeout * 1000,
                    ) as login_response_info:
                        await page.locator(self.login_button_selector).click(force=True)

                    login_response = await login_response_info.value
                    login_json = await login_response.json()
                    self.log.info(
                        "[会话模块] 页面登录接口返回 - %s",
                        json.dumps(login_json, ensure_ascii=False)[:200],
                    )
                    if login_json.get("code") not in (0, None):
                        raise RuntimeError(login_json.get("msg") or "页面登录失败")

                    try:
                        await page.wait_for_url(
                            "**/dashboard*",
                            timeout=self.timeout * 1000,
                        )
                    except PlaywrightTimeoutError:
                        self.log.warning("[会话模块] 页面未自动跳转 dashboard，尝试主动打开")
                        await page.goto(
                            self.dashboard_url,
                            wait_until="domcontentloaded",
                            timeout=self.timeout * 1000,
                        )

                    login_state = await self.get_page_login_state(page)
                    if (
                        not login_state.get("is_logged_in")
                        and login_state.get("local_token")
                        and self.context is not None
                    ):
                        self.log.warning("[会话模块] 页面登录成功但未写入 token cookie，执行浏览器上下文补写")
                        await self.context.add_cookies(
                            self.build_cookies(login_state["local_token"])
                        )
                        await page.goto(
                            self.dashboard_url,
                            wait_until="domcontentloaded",
                            timeout=self.timeout * 1000,
                        )
                        login_state = await self.get_page_login_state(page)

                    if not login_state.get("is_logged_in"):
                        raise RuntimeError("浏览器自动化登录后未检测到有效登录态")

                    self.log.info("[会话模块] 浏览器自动化登录成功 - URL: %s", page.url)
                    return build_success_response(
                        {
                            "mode": "browser",
                            "current_url": page.url,
                            "local_token": login_state.get("local_token", ""),
                        }
                    )
                except (
                    PlaywrightTimeoutError,
                    PlaywrightError,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    self.log.error(
                        "[会话模块] 浏览器自动化登录失败 - 重试: %s/%s | 错误: %s",
                        attempt,
                        self.retry,
                        exc,
                    )
                    if attempt >= self.retry:
                        return build_fail_response(str(exc))
                    await asyncio.sleep(2 ** attempt)

            return build_fail_response("浏览器自动化登录失败")

    async def open_semrush_and_collect_cookie(self) -> dict[str, Any]:
        """
        打开 Semrush 页面并抓取 wmx_business

        Returns:
            dict[str, Any]: 统一响应，包含 wmx_business
        """

        try:
            await self.start()
            if self.context is None:
                raise RuntimeError("固定浏览器尚未启动")

            async with self.browser_lock:
                page = await self.ensure_dashboard_page(force_reload=True)
                login_state = await self.get_page_login_state(page)
                self.log.info(
                    "[会话模块] 当前页面登录态检查 - is_logged_in: %s | url: %s",
                    login_state.get("is_logged_in"),
                    login_state.get("href"),
                )
                if not login_state.get("is_logged_in"):
                    raise RuntimeError("当前浏览器上下文未检测到有效登录态")

                semrush = page.get_by_text(self.semrush_entry_text, exact=True)
                await semrush.wait_for(timeout=self.timeout * 1000)
                self.log.info("[模拟操作] 已找到 Semrush 入口，准备点击")
                async with page.expect_popup(timeout=self.timeout * 1000) as popup_info:
                    await semrush.click()

                new_page = await popup_info.value
                await new_page.wait_for_load_state(
                    "networkidle",
                    timeout=self.timeout * 1000,
                )
                await new_page.wait_for_timeout(5000)
                self.log.info("[模拟操作] Semrush 新页面已加载 - URL: %s", new_page.url)
                for _ in range(30):
                    cookies = await self.context.cookies()
                    for cookie in cookies:
                        if cookie.get("name") == "wmx_business":
                            self.log.info("[会话模块] wmx_business 抓取成功")
                            return build_success_response(
                                {
                                    "wmx_business": cookie.get("value"),
                                    "cookie_domain": cookie.get("domain"),
                                    "expires": cookie.get("expires"),
                                    "semrush_url": new_page.url,
                                }
                            )
                    await new_page.wait_for_timeout(1000)

                cookie_names = [
                    f"{cookie.get('name')}@{cookie.get('domain')}"
                    for cookie in await self.context.cookies()
                ]
                raise RuntimeError(
                    f"Semrush 页面中未找到 wmx_business，当前 Cookie: {cookie_names}"
                )
        except Exception as exc:
            self.log.error("[模拟操作] Semrush 页面 Cookie 抓取失败 - 错误: %s", exc)
            return build_fail_response(str(exc))

    async def get_wmx_business_value(self) -> dict[str, Any]:
        """
        获取 wmx_business 值

        Returns:
            dict[str, Any]: 统一响应，包含 wmx_business
        """

        try:
            await self.start()
            if self.context is None:
                raise RuntimeError("固定浏览器尚未启动")

            existing_cookies = await self.context.cookies()
            for cookie in existing_cookies:
                if cookie.get("name") == "wmx_business":
                    self.log.info("[会话模块] 读取到现有 wmx_business")
                    return build_success_response(
                        {
                            "wmx_business": cookie.get("value"),
                            "cookie_domain": cookie.get("domain"),
                            "expires": cookie.get("expires"),
                        }
                    )

            self.log.info("[会话模块] 当前上下文无 wmx_business，开始执行登录链路")
            login_response = await self.ensure_browser_login()
            if login_response.get("code") != 200:
                return login_response

            return await self.open_semrush_and_collect_cookie()
        except Exception as exc:
            self.log.error("[会话模块] 获取 wmx_business 失败 - 错误: %s", exc)
            return build_fail_response(str(exc))

    async def get_status(self) -> dict[str, Any]:
        """
        获取服务状态

        Returns:
            dict[str, Any]: 统一响应，包含当前状态
        """

        await self.start()
        cookie_response = await self.get_wmx_business_value()
        current_url = ""
        if self.dashboard_page is not None and not self.dashboard_page.is_closed():
            current_url = self.dashboard_page.url

        return build_success_response(
            {
                "started": self.context is not None,
                "current_url": current_url,
                "profile_dir": str(self.user_data_dir),
                "has_wmx_business": cookie_response.get("code") == 200,
                "wmx_business": cookie_response.get("data", {}).get("wmx_business"),
            }
        )

    async def refresh_dashboard(self) -> dict[str, Any]:
        """
        刷新 dashboard 页面并重新获取 wmx_business

        Returns:
            dict[str, Any]: 统一响应
        """

        try:
            await self.start()
            async with self.browser_lock:
                await self.ensure_dashboard_page(force_reload=True)
                if self.dashboard_page is not None:
                    await self.dashboard_page.wait_for_timeout(1500)
            self.log.info("[请求模块] dashboard 刷新完成，开始重新获取 cookie")
            return await self.get_wmx_business_value()
        except Exception as exc:
            self.log.error("[请求模块] dashboard 刷新失败 - 错误: %s", exc)
            return build_fail_response(str(exc))

    async def stop(self) -> None:
        """
        关闭固定浏览器服务

        Returns:
            None
        """

        async with self.browser_lock:
            self.log.info("[主流程] 开始关闭固定浏览器服务")
            if self.context is not None:
                await self.context.close()
                self.context = None
            if self.playwright is not None:
                await self.playwright.stop()
                self.playwright = None
            self.dashboard_page = None
            self.sign_page = None
            self.log.info("[主流程] 固定浏览器服务已关闭")


async def main() -> None:
    """
    本地调试入口

    Returns:
        None
    """

    service = SemrushBrowserTokenService()
    try:
        result = await service.get_wmx_business_value()
        print(json.dumps(result, ensure_ascii=False))
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
