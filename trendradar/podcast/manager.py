# coding=utf-8
"""
播客流程管理器 (Podcast Manager Module)

协调正文拉取、AI 总结、音频生成的完整流程。
负责与存储系统交互，上传音频到 S3 并返回播放链接。

使用示例:
    manager = PodcastManager(ctx)
    audio_urls = manager.generate_podcasts(stats, title_info)
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .content_fetcher import ContentFetcher, FetchResult
from .summarizer import NewsSummarizer, SummaryResult
from .audio_generator import AudioGenerator, AudioResult


@dataclass
class PodcastResult:
    """
    播客生成结果数据类
    
    Attributes:
        keyword: 关键词
        summary: 摘要文本
        audio_url: 音频播放 URL
        audio_local_path: 本地音频文件路径
        article_count: 相关文章数量
        success: 是否成功
        error: 错误信息（如果失败）
        steps_completed: 完成的步骤列表
    """
    keyword: str
    summary: str = ""
    audio_url: str = ""
    audio_local_path: str = ""
    article_count: int = 0
    success: bool = False
    error: str = ""
    steps_completed: List[str] = field(default_factory=list)


class PodcastManager:
    """
    播客流程管理器
    
    协调整个播客生成流程：
    1. 从统计数据中提取需要生成播客的关键词和文章
    2. 使用 Jina AI 拉取文章正文
    3. 使用 LLM 生成播客摘要
    4. 使用 TTS 生成音频
    5. 上传音频到 S3 存储
    6. 返回可在飞书播放的音频链接
    
    Attributes:
        ctx: 应用上下文（AppContext）
        config: 配置字典
        content_fetcher: 正文拉取器
        summarizer: 摘要生成器
        audio_generator: 音频生成器
    """
    
    def __init__(self, ctx: Any):
        """
        初始化播客管理器
        
        Args:
            ctx: 应用上下文（AppContext），包含配置和存储管理器
        """
        self.ctx = ctx
        self.config = ctx.config
        
        # 获取播客配置
        podcast_config = self.config.get("PODCAST", {})
        
        # 初始化正文拉取器
        jina_config = podcast_config.get("JINA", {})
        self.content_fetcher = ContentFetcher(
            api_key=jina_config.get("API_KEY") or os.environ.get("JINA_API_KEY"),
            api_url=jina_config.get("API_URL"),
            proxy_url=self.config.get("DEFAULT_PROXY") if self.config.get("USE_PROXY") else None,
        )
        
        # 初始化摘要生成器
        llm_config = podcast_config.get("LLM", {})
        self.summarizer = NewsSummarizer(
            provider=llm_config.get("PROVIDER", ""),
            api_key=llm_config.get("API_KEY"),
            model=llm_config.get("MODEL"),
            api_url=llm_config.get("API_URL"),
            proxy_url=self.config.get("DEFAULT_PROXY") if self.config.get("USE_PROXY") else None,
        )
        
        # 初始化音频生成器
        tts_config = podcast_config.get("TTS", {})
        self.audio_generator = AudioGenerator(
            provider=tts_config.get("PROVIDER", ""),
            api_key=tts_config.get("API_KEY"),
            voice=tts_config.get("VOICE"),
            api_url=tts_config.get("API_URL"),
            proxy_url=self.config.get("DEFAULT_PROXY") if self.config.get("USE_PROXY") else None,
            audio_format=podcast_config.get("AUDIO_FORMAT", "mp3"),
            output_dir=podcast_config.get("OUTPUT_DIR", "output/podcast"),
        )
        
        # 其他配置
        self.max_articles_per_keyword = podcast_config.get("MAX_ARTICLES_PER_KEYWORD", 5)
        self.max_keywords = podcast_config.get("MAX_KEYWORDS", 10)
        self.fetch_delay = podcast_config.get("FETCH_DELAY", 0.5)
    
    def _extract_articles_from_stats(
        self,
        stats: List[Dict],
        title_info: Dict,
    ) -> Dict[str, List[Dict]]:
        """
        从统计数据中提取关键词和相关文章
        
        Args:
            stats: 统计数据列表（来自频率词分析）
            title_info: 标题详情信息
            
        Returns:
            Dict[str, List[Dict]]: 关键词到文章列表的映射
        """
        keyword_articles = {}
        
        # 限制关键词数量
        stats_to_process = stats[:self.max_keywords]
        
        for stat in stats_to_process:
            keyword = stat.get("word", "")
            titles_data = stat.get("titles", [])
            
            if not keyword or not titles_data:
                continue
            
            articles = []
            for title_data in titles_data[:self.max_articles_per_keyword]:
                title = title_data.get("title", "")
                url = title_data.get("url", "")
                source = title_data.get("source", "")
                
                if title and url:
                    articles.append({
                        "title": title,
                        "url": url,
                        "source": source,
                    })
            
            if articles:
                keyword_articles[keyword] = articles
        
        return keyword_articles
    
    def _fetch_contents(
        self,
        keyword_articles: Dict[str, List[Dict]],
    ) -> Dict[str, List[Dict]]:
        """
        拉取所有文章的正文内容
        
        Args:
            keyword_articles: 关键词到文章列表的映射
            
        Returns:
            Dict[str, List[Dict]]: 关键词到带内容的文章列表的映射
        """
        result = {}
        
        for keyword, articles in keyword_articles.items():
            print(f"\n📖 拉取「{keyword}」相关文章正文...")
            
            articles_with_content = []
            for article in articles:
                url = article.get("url", "")
                if not url:
                    continue
                
                fetch_result = self.content_fetcher.fetch_content(url)
                
                if fetch_result.success:
                    articles_with_content.append({
                        "title": article.get("title", fetch_result.title),
                        "url": url,
                        "content": fetch_result.content,
                        "source": article.get("source", ""),
                    })
                    print(f"  ✅ {article.get('title', '')[:30]}...")
                else:
                    print(f"  ❌ {article.get('title', '')[:30]}: {fetch_result.error}")
                
                # 添加延迟
                time.sleep(self.fetch_delay)
            
            if articles_with_content:
                result[keyword] = articles_with_content
        
        return result
    
    def _upload_to_storage(self, local_path: str) -> str:
        """
        上传音频文件到存储
        
        优先使用 S3 存储，如果未配置则使用免费临时托管服务 Litterbox。
        
        Args:
            local_path: 本地文件路径
            
        Returns:
            str: 远程 URL（如果上传成功）
        """
        # 1. 优先尝试 S3 存储
        storage_manager = getattr(self.ctx, 'storage_manager', None)
        if storage_manager and hasattr(storage_manager, 'upload_file'):
            try:
                remote_path = f"podcast/{Path(local_path).name}"
                url = storage_manager.upload_file(local_path, remote_path)
                if url:
                    return url
            except Exception as e:
                print(f"  ⚠️ S3 上传失败: {e}")
        
        # 2. 备用方案：使用 Litterbox 免费临时托管（24小时有效）
        return self._upload_to_litterbox(local_path)
    
    def _upload_to_litterbox(self, local_path: str, expiry: str = "24h") -> str:
        """
        上传文件到 Litterbox（catbox.moe 的临时存储服务）
        
        Litterbox 是一个免费的临时文件托管服务，无需注册。
        支持的有效期: 1h, 12h, 24h, 72h
        
        Args:
            local_path: 本地文件路径
            expiry: 有效期（默认 24h）
            
        Returns:
            str: 文件的公开 URL，失败返回空字符串
        """
        import requests
        
        litterbox_api = "https://litterbox.catbox.moe/resources/internals/api.php"
        
        try:
            with open(local_path, 'rb') as f:
                files = {
                    'fileToUpload': (Path(local_path).name, f, 'audio/mpeg')
                }
                data = {
                    'reqtype': 'fileupload',
                    'time': expiry  # 1h, 12h, 24h, 72h
                }
                
                response = requests.post(
                    litterbox_api,
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200 and response.text.startswith('https://'):
                    url = response.text.strip()
                    print(f"    📤 已上传到临时存储 (24h有效): {url}")
                    return url
                else:
                    print(f"  ⚠️ Litterbox 上传失败: {response.text[:100]}")
                    return ""
                    
        except Exception as e:
            print(f"  ⚠️ Litterbox 上传出错: {e}")
            return ""
    
    def generate_podcasts(
        self,
        stats: List[Dict],
        title_info: Dict,
    ) -> Dict[str, PodcastResult]:
        """
        生成播客的主流程
        
        完整流程：
        1. 提取关键词和文章
        2. 拉取正文
        3. 生成摘要
        4. 生成音频
        5. 上传存储
        
        Args:
            stats: 统计数据列表
            title_info: 标题详情信息
            
        Returns:
            Dict[str, PodcastResult]: 关键词到播客结果的映射
        """
        results = {}
        
        # 检查是否启用播客功能
        if not self.config.get("PODCAST", {}).get("ENABLED", False):
            print("📻 播客功能未启用，跳过生成")
            return results
        
        print("\n" + "=" * 50)
        print("🎙️ 开始生成热点播客")
        print("=" * 50)
        
        # 步骤 1: 提取关键词和文章
        print("\n📋 步骤 1/4: 提取关键词和文章...")
        keyword_articles = self._extract_articles_from_stats(stats, title_info)
        
        if not keyword_articles:
            print("  ⚠️ 没有可用的文章，跳过播客生成")
            return results
        
        print(f"  ✅ 提取到 {len(keyword_articles)} 个关键词")
        
        # 步骤 2: 拉取正文
        print("\n📖 步骤 2/4: 拉取文章正文...")
        keyword_contents = self._fetch_contents(keyword_articles)
        
        if not keyword_contents:
            print("  ⚠️ 正文拉取失败，跳过播客生成")
            return results
        
        # 步骤 3: 生成摘要
        print("\n📝 步骤 3/4: 生成 AI 摘要...")
        summaries = {}
        
        for keyword, articles in keyword_contents.items():
            result = PodcastResult(
                keyword=keyword,
                article_count=len(articles),
            )
            result.steps_completed.append("fetch_content")
            
            # 生成摘要
            summary_result = self.summarizer.summarize(keyword, articles)
            
            if summary_result.success:
                summaries[keyword] = summary_result.summary
                result.summary = summary_result.summary
                result.steps_completed.append("summarize")
                print(f"  ✅ 「{keyword}」摘要生成成功")
            else:
                result.error = f"摘要生成失败: {summary_result.error}"
                print(f"  ❌ 「{keyword}」摘要生成失败: {summary_result.error}")
            
            results[keyword] = result
        
        # 步骤 4: 生成音频
        print("\n🎵 步骤 4/4: 生成播客音频...")
        
        for keyword, summary_text in summaries.items():
            result = results[keyword]
            
            # 生成音频
            audio_result = self.audio_generator.generate(summary_text, keyword)
            
            if audio_result.success:
                result.audio_local_path = audio_result.local_path
                result.steps_completed.append("generate_audio")
                print(f"  ✅ 「{keyword}」音频生成成功: {audio_result.local_path}")
                
                # 尝试上传到远程存储
                remote_url = self._upload_to_storage(audio_result.local_path)
                if remote_url:
                    result.audio_url = remote_url
                    result.steps_completed.append("upload")
                    print(f"    📤 已上传: {remote_url}")
                
                result.success = True
            else:
                result.error = f"音频生成失败: {audio_result.error}"
                print(f"  ❌ 「{keyword}」音频生成失败: {audio_result.error}")
        
        # 统计结果
        success_count = sum(1 for r in results.values() if r.success)
        print("\n" + "=" * 50)
        print(f"🎙️ 播客生成完成: {success_count}/{len(results)} 成功")
        print("=" * 50)
        
        return results
    
    def get_audio_urls(
        self,
        results: Dict[str, PodcastResult],
    ) -> Dict[str, str]:
        """
        从结果中提取音频 URL
        
        Args:
            results: 播客生成结果
            
        Returns:
            Dict[str, str]: 关键词到音频 URL 的映射
        """
        return {
            keyword: result.audio_url
            for keyword, result in results.items()
            if result.success and result.audio_url
        }
