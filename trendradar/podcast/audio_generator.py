# coding=utf-8
"""
音频生成模块 (Audio Generator Module)

将摘要文本转换为播客音频。
支持多种 TTS 服务：
- Edge TTS（免费，推荐中文场景）
- OpenAI TTS
- Azure Speech Service
- 302.AI

使用示例:
    generator = AudioGenerator(config)
    audio_path = generator.generate("关于 AI 的热点资讯...", "AI")
"""

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests

# Edge TTS 支持（可选依赖）
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


@dataclass
class AudioResult:
    """
    音频生成结果数据类
    
    Attributes:
        keyword: 关键词
        local_path: 本地文件路径
        remote_url: 远程存储 URL（上传后）
        duration: 音频时长（秒）
        success: 是否成功
        error: 错误信息（如果失败）
    """
    keyword: str
    local_path: str = ""
    remote_url: str = ""
    duration: float = 0.0
    success: bool = False
    error: str = ""


class BaseTTSGenerator(ABC):
    """
    TTS 生成器基类
    
    定义 TTS 生成的标准接口，不同的服务实现此基类。
    """
    
    @abstractmethod
    def generate(
        self,
        text: str,
        keyword: str,
        output_dir: str,
    ) -> AudioResult:
        """
        生成音频文件
        
        Args:
            text: 要转换的文本
            keyword: 关键词（用于命名文件）
            output_dir: 输出目录
            
        Returns:
            AudioResult: 音频生成结果
        """
        pass


class AudioGenerator(BaseTTSGenerator):
    """
    音频生成器
    
    将文本转换为语音音频，支持多种 TTS 服务提供商。
    
    Attributes:
        provider: TTS 服务提供商（edge/openai/azure/302ai）
        api_key: API 密钥（Edge TTS 不需要）
        voice: 语音类型
        api_url: API 端点
        audio_format: 音频格式（mp3/wav/etc）
    """
    
    # 默认配置
    DEFAULT_AUDIO_FORMAT = "mp3"
    DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"  # Edge TTS 默认中文语音（晓晓）
    
    # Edge TTS 中文语音列表
    EDGE_TTS_VOICES = {
        "xiaoxiao": "zh-CN-XiaoxiaoNeural",      # 晓晓（女声，自然亲切）
        "yunxi": "zh-CN-YunxiNeural",            # 云希（男声，专业播报）
        "xiaoyi": "zh-CN-XiaoyiNeural",          # 晓依（女声，温柔）
        "yunjian": "zh-CN-YunjianNeural",        # 云健（男声，新闻播报）
        "yunxia": "zh-CN-YunxiaNeural",          # 云夏（女声，活泼）
        "yunyang": "zh-CN-YunyangNeural",        # 云扬（男声，专业）
    }
    
    def __init__(
        self,
        provider: str = "",
        api_key: Optional[str] = None,
        voice: str = "",
        api_url: Optional[str] = None,
        proxy_url: Optional[str] = None,
        audio_format: str = "mp3",
        output_dir: str = "output/podcast",
    ):
        """
        初始化音频生成器
        
        Args:
            provider: TTS 服务提供商（edge/openai/azure/302ai）
            api_key: API 密钥（Edge TTS 不需要）
            voice: 语音类型
            api_url: API 端点（可选）
            proxy_url: 代理服务器 URL（可选）
            audio_format: 音频格式（默认 mp3）
            output_dir: 输出目录
        """
        self.provider = provider.lower() if provider else ""
        self.api_key = api_key or ""
        self.voice = voice or self.DEFAULT_VOICE
        self.api_url = api_url or ""
        self.proxy_url = proxy_url
        self.audio_format = audio_format or self.DEFAULT_AUDIO_FORMAT
        self.output_dir = output_dir
        
        # 根据 provider 设置默认值
        self._setup_provider_defaults()
    
    def _setup_provider_defaults(self):
        """根据 provider 设置默认的 API 端点和配置"""
        if self.provider == "edge":
            # Edge TTS 免费，无需 API Key
            # 如果 voice 是简写，转换为完整名称
            if self.voice.lower() in self.EDGE_TTS_VOICES:
                self.voice = self.EDGE_TTS_VOICES[self.voice.lower()]
            elif not self.voice.startswith("zh-"):
                # 默认使用晓晓
                self.voice = self.DEFAULT_VOICE
        elif self.provider == "openai":
            self.api_url = self.api_url or "https://api.openai.com/v1/audio/speech"
            self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
            self.voice = self.voice or "alloy"
        elif self.provider == "302ai":
            # 302.AI 使用 OpenAI 兼容接口
            self.api_url = self.api_url or "https://api.302.ai/v1/audio/speech"
            self.api_key = self.api_key or os.environ.get("AI302_API_KEY", "")
            self.voice = self.voice or "alloy"
        elif self.provider == "azure":
            # Azure Speech Service 需要特殊配置
            self.api_key = self.api_key or os.environ.get("AZURE_SPEECH_KEY", "")
        # 其他 provider 需要用户提供完整配置
    
    def _ensure_output_dir(self) -> Path:
        """确保输出目录存在"""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def _generate_filename(self, keyword: str) -> str:
        """
        生成音频文件名
        
        Args:
            keyword: 关键词
            
        Returns:
            文件名（不含路径）
        """
        # 使用时间戳确保唯一性
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 清理关键词中的特殊字符
        safe_keyword = "".join(c for c in keyword if c.isalnum() or c in "_ -")
        safe_keyword = safe_keyword[:30]  # 限制长度
        
        return f"podcast_{safe_keyword}_{timestamp}.{self.audio_format}"
    
    def generate(
        self,
        text: str,
        keyword: str,
        output_dir: Optional[str] = None,
    ) -> AudioResult:
        """
        生成音频文件
        
        Args:
            text: 要转换的文本
            keyword: 关键词（用于命名文件）
            output_dir: 输出目录（可选，默认使用初始化时的配置）
            
        Returns:
            AudioResult: 音频生成结果
        """
        # 检查配置
        if not self.provider:
            return AudioResult(
                keyword=keyword,
                success=False,
                error="未配置 TTS 服务提供商（provider）"
            )
        
        # Edge TTS 不需要 API Key，其他提供商需要
        if self.provider != "edge" and not self.api_key:
            return AudioResult(
                keyword=keyword,
                success=False,
                error=f"未配置 {self.provider.upper()} API 密钥"
            )
        
        if not text or not text.strip():
            return AudioResult(
                keyword=keyword,
                success=False,
                error="文本内容为空"
            )
        
        # 确定输出目录
        final_output_dir = output_dir or self.output_dir
        output_path = Path(final_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        filename = self._generate_filename(keyword)
        file_path = output_path / filename
        
        # 根据 provider 调用相应的 TTS API
        try:
            if self.provider == "edge":
                # Edge TTS 使用异步接口
                success = self._call_edge_tts(text, str(file_path))
                if success:
                    return AudioResult(
                        keyword=keyword,
                        local_path=str(file_path),
                        success=True
                    )
                else:
                    return AudioResult(
                        keyword=keyword,
                        success=False,
                        error="Edge TTS 生成失败"
                    )
            elif self.provider in ["openai", "302ai"]:
                audio_data = self._call_openai_tts(text)
            elif self.provider == "azure":
                audio_data = self._call_azure_tts(text)
            else:
                return AudioResult(
                    keyword=keyword,
                    success=False,
                    error=f"不支持的 TTS 服务提供商: {self.provider}"
                )
            
            # 保存音频文件
            with open(file_path, "wb") as f:
                f.write(audio_data)
            
            return AudioResult(
                keyword=keyword,
                local_path=str(file_path),
                success=True
            )
            
        except Exception as e:
            return AudioResult(
                keyword=keyword,
                success=False,
                error=str(e)
            )
    
    def _call_edge_tts(self, text: str, output_path: str) -> bool:
        """
        调用 Edge TTS 生成音频
        
        使用微软 Edge 浏览器的神经网络语音合成服务（免费）。
        
        Args:
            text: 要转换的文本
            output_path: 输出文件路径
            
        Returns:
            bool: 是否成功
        """
        if not EDGE_TTS_AVAILABLE:
            raise ImportError(
                "Edge TTS 未安装，请运行: pip install edge-tts"
            )
        
        async def _generate():
            """异步生成音频"""
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
        
        try:
            # 运行异步任务
            asyncio.run(_generate())
            return True
        except Exception as e:
            print(f"  ❌ Edge TTS 生成失败: {e}")
            return False
    
    def _call_openai_tts(self, text: str) -> bytes:
        """
        调用 OpenAI TTS API
        
        Args:
            text: 要转换的文本
            
        Returns:
            bytes: 音频数据
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": self.voice,
            "response_format": self.audio_format,
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
            timeout=120,  # TTS 可能需要较长时间
        )
        
        response.raise_for_status()
        return response.content
    
    def _call_azure_tts(self, text: str) -> bytes:
        """
        调用 Azure Speech Service TTS API
        
        Args:
            text: 要转换的文本
            
        Returns:
            bytes: 音频数据
            
        注意：Azure TTS 需要额外配置 region 和 SSML 格式
        """
        # Azure TTS 实现预留
        raise NotImplementedError(
            "Azure TTS 支持尚未实现，请使用 openai 或 302ai provider"
        )
    
    def generate_batch(
        self,
        summaries: Dict[str, str],
        output_dir: Optional[str] = None,
    ) -> Dict[str, AudioResult]:
        """
        批量生成多个关键词的音频
        
        Args:
            summaries: 关键词到摘要文本的映射
            output_dir: 输出目录（可选）
            
        Returns:
            Dict[str, AudioResult]: 关键词到音频结果的映射
        """
        results = {}
        total = len(summaries)
        
        print(f"🎙️ 开始生成音频，共 {total} 个关键词")
        
        for i, (keyword, text) in enumerate(summaries.items(), 1):
            print(f"  [{i}/{total}] 正在生成「{keyword}」的音频...")
            
            result = self.generate(text, keyword, output_dir)
            results[keyword] = result
            
            if result.success:
                print(f"    ✅ 成功: {result.local_path}")
            else:
                print(f"    ❌ 失败: {result.error}")
        
        # 统计结果
        success_count = sum(1 for r in results.values() if r.success)
        print(f"📊 音频生成完成: {success_count}/{total} 成功")
        
        return results
