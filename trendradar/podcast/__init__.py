# coding=utf-8
"""
播客模块 (Podcast Module)

提供热点新闻播客生成功能：
- 正文拉取（Jina AI）
- AI 摘要生成
- TTS 音频生成
- 播客流程管理

使用示例:
    from trendradar.podcast import PodcastManager
    
    manager = PodcastManager(config)
    audio_urls = manager.generate_podcasts(stats, title_info)
"""

from .content_fetcher import ContentFetcher
from .summarizer import NewsSummarizer
from .audio_generator import AudioGenerator
from .manager import PodcastManager

__all__ = [
    "ContentFetcher",
    "NewsSummarizer", 
    "AudioGenerator",
    "PodcastManager",
]
