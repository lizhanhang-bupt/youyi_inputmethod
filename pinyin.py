from pypinyin import lazy_pinyin, Style
from Pinyin2Hanzi import DefaultHmmParams, viterbi
import re
import math
import threading
import json
import logging
from typing import List, Dict
from collections import defaultdict
import jieba
import hashlib
from typing import Optional
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PinyinConverter:
    def __init__(self, user_id: Optional[str] = None, enable_learning: bool = True):
        """
        :param user_id: 用户唯一标识（哈希处理）
        :param enable_learning: 是否启用学习功能（默认开启）
        """
        self.hmm_params = self._load_enhanced_hmm('training_data.txt')  # 指定训练数据路径
        self.pinyin_pattern = re.compile(r'^[a-z]+( [a-z]+)*$')  # 更严格的拼音正则
        self._cache = {}
        self._init_fallback_strategy()
        self.enable_learning = enable_learning
        self.user_dict = self._load_user_dict(user_id)
        self.privacy_lock = threading.Lock()  # 线程安全锁
        jieba.initialize()

    def _split_pinyin(self, pinyin_text: str) -> List[str]:
        """ 使用jieba代替原来的segmenter """
        try:
            # 使用jieba的精确模式分割
            return list(jieba.cut(pinyin_text, cut_all=False))
        except Exception as e:
            logger.error(f"分词失败: {str(e)}")
            return self._heuristic_segment(pinyin_text)  # 降级策略

    def _load_user_dict(self, user_id: Optional[str]) -> dict:
        """安全加载用户词典"""
        if not user_id or not self.enable_learning:
            return defaultdict(list)

        # 生成加密文件名
        filename = f"user_{hashlib.sha256(user_id.encode()).hexdigest()[:16]}.dict"
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return defaultdict(list)

    def _save_user_dict(self):
        """加密保存用户词典"""
        if not self.user_id or not self.enable_learning:
            return

        filename = f"user_{hashlib.sha256(self.user_id.encode()).hexdigest()[:16]}.dict"
        with self.privacy_lock:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.user_dict, f, ensure_ascii=False)

    def update_learning_setting(self, enable: bool):
        """隐私设置开关"""
        with self.privacy_lock:
            old_setting = self.enable_learning
            self.enable_learning = enable

            # 关闭时立即清除内存中的用户数据
            if old_setting and not enable:
                self.user_dict.clear()

    def _update_user_model(self, pinyin: str, selected_text: str):
        """安全更新用户模型"""
        if not self.enable_learning:
            return

        with self.privacy_lock:
            # 限制最大存储条目
            if len(self.user_dict[pinyin]) < 100:  # 防止内存溢出
                # 增加权重（存在则提升排名）
                entries = [t for t in self.user_dict[pinyin] if t[0] != selected_text]
                entries.insert(0, (selected_text, 1.0))
                self.user_dict[pinyin] = entries[:10]  # 保留前10个
                self._save_user_dict()

    def _load_enhanced_hmm(self, train_file: str):
        """ 从训练数据中学习HMM参数 """
        params = DefaultHmmParams()

        # 从训练文件统计转移概率
        transition_counts = self._train_transition_probs(train_file)

        # 转换计数为对数概率
        for current_char, next_chars in transition_counts.items():
            total = sum(next_chars.values())
            # 使用拉普拉斯平滑避免零概率
            for next_char in next_chars:
                next_chars[next_char] += 1
                total += 1
            # 更新HMM参数
            params.transition_dict[current_char] = {
                next_char: math.log(count / total)
                for next_char, count in next_chars.items()
            }
        return params

    def _train_transition_probs(self, file_path: str):
        """ 统计汉字转移频次 """
        transition_counts = defaultdict(lambda: defaultdict(int))
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = re.sub(r'\s+', '', f.read())  # 移除所有空白字符
                text = re.sub(r'[^\u4e00-\u9fff]', '', text)  # 只保留汉字

                # 统计相邻汉字转移
                for i in range(len(text) - 1):
                    current = text[i]
                    next_char = text[i + 1]
                    transition_counts[current][next_char] += 1
        except FileNotFoundError:
            logger.warning(f"训练文件 {file_path} 未找到，使用默认HMM参数")
        return transition_counts

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
        results = []  # 初始化results变量
        pinyin_text = pinyin_text.strip().replace(' ', '')

        if not self.is_valid_pinyin(pinyin_text):
            logger.warning(f"非法拼音输入: {pinyin_text}")
            return []

        if pinyin_text in self._cache:
            return self._cache[pinyin_text]

        try:
            pinyin_list = self._split_pinyin(pinyin_text)
            raw_results = self._viterbi_decode(pinyin_list, context)
            results = self._post_process(raw_results, top_k) if raw_results else []

            if self.enable_learning and pinyin_text in self.user_dict:
                results = self._apply_user_preference(results, pinyin_text)

            if not results:
                results = self._fallback_strategy(pinyin_text)

            return results
        except Exception as e:
            logger.error(f"转换失败: {str(e)}", exc_info=True)
            results = self._fallback_strategy(pinyin_text)
            return results
        finally:
            if results:  # 只有当results被赋值时才缓存
                self._cache[pinyin_text] = results

    def _apply_user_preference(self, results: List[Dict], pinyin: str) -> List[Dict]:
        """应用用户个性化排序"""
        user_entries = {text: weight for text, weight in self.user_dict[pinyin]}

        # 提升用户偏好项的得分
        for res in results:
            if res['text'] in user_entries:
                res['score'] *= user_entries[res['text']] * 2  # 权重加倍

        return sorted(results, key=lambda x: -x['score'])

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

class PrivacyController:
    @staticmethod
    def anonymize_input(text: str) -> str:
        """数据脱敏处理"""
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def encrypt_data(data: dict, key: bytes) -> bytes:
        """AES加密用户数据"""
        # 实现实际的加密逻辑（示例）
        from Crypto.Cipher import AES
        cipher = AES.new(key, AES.MODE_EAX)
        nonce = cipher.nonce
        ciphertext, tag = cipher.encrypt_and_digest(json.dumps(data).encode())
        return nonce + tag + ciphertext

    @classmethod
    def create_consent_dialog(cls):
        """生成隐私协议对话框（伪代码）"""
        print("""
        隐私保护声明：
        1. 用户输入数据仅用于改进输入体验
        2. 所有数据经过加密处理
        3. 可随时关闭学习功能
        是否同意？(y/n)
        """)
        choice = input().strip().lower()
        return choice == 'y'
if __name__ == "__main__":
    # 用户首次使用
    user_id = "user@example.com"  # 实际应使用不可逆哈希值

    # 隐私协议确认
    if PrivacyController.create_consent_dialog():
        converter = PinyinConverter(user_id=user_id)
    else:
        converter = PinyinConverter(user_id=None)

    # 正常使用流程
    result = converter.convert("zhangsan")
    print(result)

    # 用户选择后更新设置
    converter.update_learning_setting(False)  # 关闭学习

    # 查看隐私数据（示例）
    print("脱敏后的用户输入示例:", PrivacyController.anonymize_input("张三"))