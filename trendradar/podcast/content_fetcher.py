# coding=utf-8
"""
正文拉取模块 (Content Fetcher Module)

使用 Jina AI Reader API (r.jina.ai) 从 URL 获取网页正文内容。
该模块负责将热点新闻的原始 URL 转换为 LLM 友好的文本格式。

Get your Jina AI API key for free: https://jina.ai/?sui=apikey

使用示例:
    fetcher = ContentFetcher()
    content = fetcher.fetch_content("https://example.com/news/12345")
"""

import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import requests


@dataclass
class FetchResult:
    """
    正文拉取结果数据类
    
    Attributes:
        url: 原始 URL
        title: 页面标题
        content: 正文内容（Markdown 格式）
        description: 页面描述
        success: 是否成功
        error: 错误信息（如果失败）
        tokens: 消耗的 token 数量
    """
    url: str
    title: str = ""
    content: str = ""
    description: str = ""
    success: bool = False
    error: str = ""
    tokens: int = 0


class ContentFetcher:
    """
    Jina AI 正文拉取器
    
    使用 Jina Reader API 从网页 URL 提取结构化的正文内容，
    输出格式为 Markdown，适合后续 LLM 处理。
    
    Attributes:
        api_url: Jina Reader API 端点
        api_key: Jina AI API 密钥
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
    """
    
    # Jina Reader API 端点
    DEFAULT_API_URL = "https://r.jina.ai/"
    
    # 默认请求头
    DEFAULT_HEADERS = {
        "Accept": "application/json",  # 必须指定 JSON 格式响应
        "Content-Type": "application/json",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 2,
        proxy_url: Optional[str] = None,
    ):
        """
        初始化正文拉取器
        
        Args:
            api_key: Jina AI API 密钥，如果不提供则从环境变量 JINA_API_KEY 读取
            api_url: API 端点 URL（可选，默认使用官方端点）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            proxy_url: 代理服务器 URL（可选）
        """
        # 从环境变量或参数获取 API 密钥
        self.api_key = api_key or os.environ.get("JINA_API_KEY", "")
        self.api_url = api_url or self.DEFAULT_API_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.proxy_url = proxy_url
        
        # 验证 API 密钥
        if not self.api_key:
            print("⚠️ 警告：未设置 JINA_API_KEY 环境变量")
            print("   请前往 https://jina.ai/?sui=apikey 获取免费 API 密钥")
    
    def _build_headers(self) -> Dict[str, str]:
        """
        构建请求头
        
        Returns:
            包含认证信息的请求头字典
        """
        headers = self.DEFAULT_HEADERS.copy()
        
        # 添加认证头
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # 可选：设置超时时间（通过 X-Timeout 头）
        headers["X-Timeout"] = str(self.timeout)
        
        # 可选：返回 Markdown 格式
        headers["X-Return-Format"] = "markdown"
        
        return headers
    
    def fetch_content(self, url: str) -> FetchResult:
        """
        从指定 URL 获取正文内容
        
        使用 Jina Reader API 将网页转换为 LLM 友好的 Markdown 格式。
        
        Args:
            url: 要抓取的网页 URL
            
        Returns:
            FetchResult: 包含正文内容的结果对象
        """
        # 检查 API 密钥
        if not self.api_key:
            return FetchResult(
                url=url,
                success=False,
                error="未配置 JINA_API_KEY，请设置环境变量或在配置中提供"
            )
        
        # 构建请求
        headers = self._build_headers()
        payload = {"url": url}
        
        # 配置代理
        proxies = None
        if self.proxy_url:
            proxies = {"http": self.proxy_url, "https": self.proxy_url}
        
        # 重试机制
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                # 发送 POST 请求
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    proxies=proxies,
                    timeout=self.timeout,
                )
                
                # 检查 HTTP 状态码
                response.raise_for_status()
                
                # 解析 JSON 响应
                result = response.json()
                
                # 检查响应状态
                if result.get("code") != 200:
                    error_msg = result.get("message", "未知错误")
                    return FetchResult(
                        url=url,
                        success=False,
                        error=f"API 返回错误: {error_msg}"
                    )
                
                # 提取数据
                data = result.get("data", {})
                
                return FetchResult(
                    url=url,
                    title=data.get("title", ""),
                    content=data.get("content", ""),
                    description=data.get("description", ""),
                    success=True,
                    tokens=data.get("usage", {}).get("tokens", 0)
                )
                
            except requests.exceptions.Timeout:
                last_error = f"请求超时（{self.timeout}秒）"
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP 错误: {e.response.status_code}"
            except requests.exceptions.RequestException as e:
                last_error = f"网络请求失败: {str(e)}"
            except Exception as e:
                last_error = f"解析响应失败: {str(e)}"
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_retries:
                wait_time = (attempt + 1) * 2  # 递增等待时间
                print(f"  ⚠️ 拉取 {url} 失败，{wait_time}秒后重试...")
                time.sleep(wait_time)
        
        # 所有重试都失败
        return FetchResult(
            url=url,
            success=False,
            error=last_error
        )
    
    def fetch_batch(
        self,
        urls: List[str],
        delay: float = 0.5,
    ) -> Dict[str, FetchResult]:
        """
        批量获取多个 URL 的正文内容
        
        Args:
            urls: URL 列表
            delay: 请求间隔时间（秒），避免触发速率限制
            
        Returns:
            Dict[str, FetchResult]: URL 到结果的映射
        """
        results = {}
        total = len(urls)
        
        print(f"📖 开始批量拉取正文，共 {total} 个 URL")
        
        for i, url in enumerate(urls, 1):
            print(f"  [{i}/{total}] 正在拉取: {url[:60]}...")
            
            result = self.fetch_content(url)
            results[url] = result
            
            if result.success:
                content_preview = result.content[:50] + "..." if len(result.content) > 50 else result.content
                print(f"    ✅ 成功，标题: {result.title[:30]}...")
            else:
                print(f"    ❌ 失败: {result.error}")
            
            # 添加延迟，避免触发 API 速率限制
            if i < total and delay > 0:
                time.sleep(delay)
        
        # 统计结果
        success_count = sum(1 for r in results.values() if r.success)
        print(f"📊 拉取完成: {success_count}/{total} 成功")
        
        return results
    
    def fetch_for_keyword(
        self,
        keyword: str,
        articles: List[Dict],
        max_articles: int = 5,
        delay: float = 0.5,
    ) -> Tuple[str, List[FetchResult]]:
        """
        为指定关键词拉取相关文章的正文
        
        Args:
            keyword: 关键词/词组
            articles: 文章列表，每个文章包含 title, url 等字段
            max_articles: 最多拉取的文章数量
            delay: 请求间隔时间（秒）
            
        Returns:
            Tuple[str, List[FetchResult]]: (关键词, 拉取结果列表)
        """
        print(f"\n🔍 拉取关键词「{keyword}」相关文章正文...")
        
        # 限制文章数量
        articles_to_fetch = articles[:max_articles]
        
        results = []
        for i, article in enumerate(articles_to_fetch, 1):
            url = article.get("url", "")
            title = article.get("title", "")
            
            if not url:
                print(f"  [{i}] ⚠️ 文章无 URL，跳过: {title[:30]}...")
                continue
            
            print(f"  [{i}/{len(articles_to_fetch)}] {title[:40]}...")
            
            result = self.fetch_content(url)
            result.title = title  # 保留原始标题
            results.append(result)
            
            if result.success:
                print(f"    ✅ 成功，内容长度: {len(result.content)} 字符")
            else:
                print(f"    ❌ 失败: {result.error}")
            
            # 添加延迟
            if i < len(articles_to_fetch) and delay > 0:
                time.sleep(delay)
        
        return keyword, results
