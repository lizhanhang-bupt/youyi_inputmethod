from pypinyin import lazy_pinyin, Style
from Pinyin2Hanzi import DefaultHmmParams, viterbi
import re
import logging
from typing import List, Dict
import hanlp

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PinyinConverter:
    def __init__(self):
        """ 加载拼音转汉字所需资源 """
        self.hmm_params = self._load_enhanced_hmm()  # 增强的HMM参数
        # 更严格的拼音正则（允许单个空格分隔）
        self.pinyin_pattern = re.compile(r'^[a-z]+( [a-z]+)*$')
        self._cache = {}
        self._init_fallback_strategy()
        self.segmenter = hanlp.load('LARGE_ALBERT_BASE')

    def _load_enhanced_hmm(self):
        """ 加载增强版HMM参数（包含更多上下文特征）"""
        params = DefaultHmmParams()
        # 示例：手动添加常见多音字转移概率
        params.transition_dict['zhang']['san'] = 0.9  # 张三
        params.transition_dict['zhang']['kai'] = 0.7  # 张开
        return params
        #后面找个模型然后用文章训练

    def _init_fallback_strategy(self):
        """ 初始化降级策略资源 """
        from pypinyin import pinyin
        self.fallback_pinyin = pinyin

    def is_valid_pinyin(self, text: str) -> bool:
        """ 增强版输入验证 """
        if not text:
            return False
        # 允许空格分隔但需符合规范
        return self.pinyin_pattern.match(text.strip()) is not None

    def convert(self, pinyin_text: str, context: str = "", top_k: int = 5) -> List[Dict]:
        """
        增强版拼音转换
        :param context: 上文内容（用于多音字消歧）
        :return: 按概率排序的候选列表
        """
        pinyin_text = pinyin_text.strip().replace(' ', '')  # 统一处理空格
        if not self.is_valid_pinyin(pinyin_text):
            logger.warning(f"非法拼音输入: {pinyin_text}")
            return []

        if pinyin_text in self._cache:
            return self._cache[pinyin_text]

        try:
            # 尝试专业分割 -> Viterbi解码
            pinyin_list = self._split_pinyin(pinyin_text)
            results = self._viterbi_decode(pinyin_list, context)

            if not results:
                # 降级策略：整词匹配
                return self._fallback_strategy(pinyin_text)

            return self._post_process(results, top_k)
        except Exception as e:
            logger.error(f"转换失败: {str(e)}", exc_info=True)
            return self._fallback_strategy(pinyin_text)
        finally:
            self._cache[pinyin_text] = results  # 缓存结果

    def _split_pinyin(self, pinyin_text: str) -> List[str]:
        """ 专业拼音分割 """
        result = self.segmenter.segment(pinyin_text)
        return self._heuristic_segment(pinyin_text)

    def _heuristic_segment(self, pinyin_text: str) -> List[str]:
        """ 基于常见音节的分割（可扩展） """
        common_syllables = {'zh', 'ch', 'sh', 'ang', 'eng', 'ing'}
        segments = []
        i = 0
        while i < len(pinyin_text):
            # 优先匹配3字符音节
            if i + 3 <= len(pinyin_text) and pinyin_text[i:i + 3] in common_syllables:
                segments.append(pinyin_text[i:i + 3])
                i += 3
            # 匹配2字符音节
            elif i + 2 <= len(pinyin_text) and pinyin_text[i:i + 2] in common_syllables:
                segments.append(pinyin_text[i:i + 2])
                i += 2
            else:
                segments.append(pinyin_text[i])
                i += 1
        return segments

    def _viterbi_decode(self, pinyin_list: List[str], context: str) -> List[Dict]:
        """ 带上下文的多音字解码 """
        # 将上下文转换为拼音特征
        context_pinyins = lazy_pinyin(context) if context else []
        return viterbi(
            pinyin_list=context_pinyins + pinyin_list,  # 组合上下文
            hmm_params=self.hmm_params,
            path_num=20,  # 获取更多候选
            log_prob=True
        )

    def _fallback_strategy(self, pinyin_text: str) -> List[Dict]:
        """ 降级策略：整词匹配 """
        try:
            # 获取所有可能的汉字组合
            candidates = self.fallback_pinyin(
                pinyin_text,
                style=Style.NORMAL,
                heteronym=True  # 启用多音字
            )
            return [{"text": ''.join(c), "score": 0.5} for c in candidates]
        except:
            return [{"text": pinyin_text, "score": 0.0}]

    def _post_process(self, results: List[Dict], top_k: int) -> List[Dict]:
        """ 后处理：去重、排序、截断 """
        seen = set()
        unique_results = []
        for res in sorted(results, key=lambda x: -x['score']):
            text = res['text']
            if text not in seen:
                seen.add(text)
                unique_results.append(res)
        return unique_results[:top_k]


if __name__ == "__main__":
    # 测试用例
    converter = PinyinConverter()

    # 正常用例
    print(converter.convert("zhangsan"))  # 张三

    # 多音字测试（依赖上下文）
    print(converter.convert("zhang", context="我姓"))  # 张
    print(converter.convert("zhang", context="头发很"))  # 长

    # 非法输入
    print(converter.convert("ni3hao"))  # []

    # 降级策略测试
    print(converter.convert("unknownpinyin"))  # 返回近似匹配