# coding=utf-8
"""
AI 摘要生成模块 (News Summarizer Module)

将拉取到的新闻正文按关键词聚合后，生成适合播客朗读的摘要文本。
该模块预留了多种 LLM 服务接口，可根据配置选择不同的提供商。

使用示例:
    summarizer = NewsSummarizer(config)
    summary = summarizer.summarize("AI", articles_content)
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


@dataclass
class SummaryResult:
    """
    摘要生成结果数据类
    
    Attributes:
        keyword: 关键词
        summary: 生成的摘要文本（适合播客朗读）
        article_count: 参与总结的文章数量
        success: 是否成功
        error: 错误信息（如果失败）
        tokens_used: 消耗的 token 数量
    """
    keyword: str
    summary: str = ""
    article_count: int = 0
    success: bool = False
    error: str = ""
    tokens_used: int = 0


class BaseSummarizer(ABC):
    """
    摘要生成器基类
    
    定义摘要生成的标准接口，不同的 LLM 服务实现此基类。
    """
    
    @abstractmethod
    def summarize(
        self,
        keyword: str,
        articles: List[Dict],
    ) -> SummaryResult:
        """
        生成关键词相关新闻的摘要
        
        Args:
            keyword: 关键词/词组
            articles: 文章列表，每篇包含 title, content 等字段
            
        Returns:
            SummaryResult: 摘要结果
        """
        pass


class NewsSummarizer(BaseSummarizer):
    """
    新闻摘要生成器
    
    将多篇新闻文章内容合并，生成适合播客朗读的摘要。
    支持多种 LLM 服务提供商（通过配置切换）。
    
    Attributes:
        provider: LLM 服务提供商（openai/302ai/other）
        api_key: API 密钥
        model: 模型名称
        api_url: API 端点
    """
    
    # 预设的系统提示词模板（生成播客风格的摘要）
    SYSTEM_PROMPT = """你是一位专业的新闻播客主持人。请根据提供的新闻内容，生成一段适合播客朗读的摘要。

要求：
1. 语言流畅自然，适合口语朗读
2. 时长控制在 30-60 秒左右（约 100-200 字）
3. 突出关键信息和热点要点
4. 避免使用难以朗读的符号和数字
5. 以「关于{keyword}的热点资讯」开头
6. 用简洁有力的语言结束

请直接输出摘要文本，不要添加额外说明。"""

    # 用户提示词模板
    USER_PROMPT_TEMPLATE = """关键词：{keyword}

相关新闻内容：
{articles_content}

请生成适合播客朗读的摘要。"""
    
    def __init__(
        self,
        provider: str = "",
        api_key: Optional[str] = None,
        model: str = "",
        api_url: Optional[str] = None,
        proxy_url: Optional[str] = None,
        max_content_length: int = 4000,
    ):
        """
        初始化摘要生成器
        
        Args:
            provider: LLM 服务提供商（openai/302ai/other）
            api_key: API 密钥
            model: 模型名称
            api_url: API 端点（可选）
            proxy_url: 代理服务器 URL（可选）
            max_content_length: 每篇文章最大内容长度
        """
        self.provider = provider.lower() if provider else ""
        self.api_key = api_key or ""
        self.model = model or ""
        self.api_url = api_url or ""
        self.proxy_url = proxy_url
        self.max_content_length = max_content_length
        
        # 根据 provider 设置默认值
        self._setup_provider_defaults()
    
    def _setup_provider_defaults(self):
        """根据 provider 设置默认的 API 端点和模型"""
        if self.provider == "openai":
            self.api_url = self.api_url or "https://api.openai.com/v1/chat/completions"
            self.model = self.model or "gpt-4o-mini"
            self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        elif self.provider == "deepseek":
            # DeepSeek 官方 API（OpenAI 兼容）
            self.api_url = self.api_url or "https://api.deepseek.com/chat/completions"
            self.model = self.model or "deepseek-chat"
            self.api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        elif self.provider == "302ai":
            # 302.AI 使用 OpenAI 兼容接口
            self.api_url = self.api_url or "https://api.302.ai/v1/chat/completions"
            self.model = self.model or "gpt-4o-mini"
            self.api_key = self.api_key or os.environ.get("AI302_API_KEY", "")
        # 其他 provider 需要用户提供完整配置
    
    def _prepare_articles_content(self, articles: List[Dict]) -> str:
        """
        准备文章内容文本
        
        将多篇文章的内容合并为一个文本，用于 LLM 输入。
        
        Args:
            articles: 文章列表
            
        Returns:
            合并后的文章内容文本
        """
        content_parts = []
        
        for i, article in enumerate(articles, 1):
            title = article.get("title", "无标题")
            content = article.get("content", "")
            
            # 截断过长的内容
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."
            
            content_parts.append(f"【新闻 {i}】{title}\n{content}")
        
        return "\n\n".join(content_parts)
    
    def summarize(
        self,
        keyword: str,
        articles: List[Dict],
    ) -> SummaryResult:
        """
        生成关键词相关新闻的摘要
        
        Args:
            keyword: 关键词/词组
            articles: 文章列表，每篇包含 title, content 等字段
            
        Returns:
            SummaryResult: 摘要结果
        """
        # 检查配置
        if not self.provider:
            return SummaryResult(
                keyword=keyword,
                success=False,
                error="未配置 LLM 服务提供商（provider）"
            )
        
        if not self.api_key:
            return SummaryResult(
                keyword=keyword,
                success=False,
                error=f"未配置 {self.provider.upper()} API 密钥"
            )
        
        if not articles:
            return SummaryResult(
                keyword=keyword,
                success=False,
                error="没有可用的文章内容"
            )
        
        # 准备内容
        articles_content = self._prepare_articles_content(articles)
        
        # 构建提示词
        system_prompt = self.SYSTEM_PROMPT.format(keyword=keyword)
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            keyword=keyword,
            articles_content=articles_content
        )
        
        # 调用 LLM API
        try:
            result = self._call_llm_api(system_prompt, user_prompt)
            
            return SummaryResult(
                keyword=keyword,
                summary=result.get("content", ""),
                article_count=len(articles),
                success=True,
                tokens_used=result.get("tokens", 0)
            )
            
        except Exception as e:
            return SummaryResult(
                keyword=keyword,
                article_count=len(articles),
                success=False,
                error=str(e)
            )
    
    def _call_llm_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict:
        """
        调用 LLM API（OpenAI 兼容格式）
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            Dict: 包含 content 和 tokens 的结果字典
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 500,
        }
        
        # 配置代理
        proxies = None
        if self.proxy_url:
            proxies = {"http": self.proxy_url, "https": self.proxy_url}
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            proxies=proxies,
            timeout=60,
        )
        
        response.raise_for_status()
        result = response.json()
        
        # 解析响应（OpenAI 格式）
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = result.get("usage", {}).get("total_tokens", 0)
        
        return {"content": content, "tokens": tokens}
    
    def summarize_batch(
        self,
        keyword_articles: Dict[str, List[Dict]],
    ) -> Dict[str, SummaryResult]:
        """
        批量生成多个关键词的摘要
        
        Args:
            keyword_articles: 关键词到文章列表的映射
            
        Returns:
            Dict[str, SummaryResult]: 关键词到摘要结果的映射
        """
        results = {}
        total = len(keyword_articles)
        
        print(f"📝 开始生成摘要，共 {total} 个关键词")
        
        for i, (keyword, articles) in enumerate(keyword_articles.items(), 1):
            print(f"  [{i}/{total}] 正在总结「{keyword}」...")
            
            result = self.summarize(keyword, articles)
            results[keyword] = result
            
            if result.success:
                print(f"    ✅ 成功，摘要长度: {len(result.summary)} 字符")
            else:
                print(f"    ❌ 失败: {result.error}")
        
        # 统计结果
        success_count = sum(1 for r in results.values() if r.success)
        print(f"📊 摘要完成: {success_count}/{total} 成功")
        
        return results
