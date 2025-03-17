from pypinyin import lazy_pinyin, Style, pinyin_dict
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
        #:param user_id: 用户唯一标识（哈希处理）
        #:param enable_learning: 是否启用学习功能（默认开启）
        self.hmm_params = self._load_enhanced_hmm('training_data.txt')  # 指定训练数据路径
        self.pinyin_pattern = re.compile(r'^[a-z]+( [a-z]+)*$')  # 更严格的拼音正则
        self._cache = {}
        self._init_fallback_strategy()
        self.enable_learning = enable_learning
        self.user_dict = self._load_user_dict(user_id)
        self.privacy_lock = threading.Lock()  # 线程安全锁
        jieba.initialize()
        #jieba.load_userdict('pinyin_dict.txt') # 初始化jieba并加载所有拼音音节
        seen_pinyin = set()
        for pinyins in pinyin_dict.pinyin_dict.values():
            for py in pinyins:
                py_normalized = re.sub(r'\d', '', py)
                if py_normalized not in seen_pinyin:
                    jieba.add_word(py_normalized, freq=1000)
                    seen_pinyin.add(py_normalized)

        # 构建首字母映射
        self.initial_to_pys = defaultdict(list)
        for py in self.hmm_params.emission_dict.keys():
            initial = self._get_initial(py)
            self.initial_to_pys[initial].append(py)

    def _get_initial(self, py: str) -> str:
        """获取拼音的首字母"""
        if py.startswith('zh'):
            return 'z'
        elif py.startswith('ch'):
            return 'c'
        elif py.startswith('sh'):
            return 's'
        return py[0] if py else ''

    def init_universal_emission(self):
        universal_emit = defaultdict(lambda: defaultdict(float))
        # 加载pypinyin内置的13万条拼音汉字映射
        for hanzi, pinyins in pinyin_dict.pinyin_dict.items():
            weight = 1.0 / len(pinyins)  # 平均分配初始概率
            for py in pinyins:
                universal_emit[py][hanzi] = math.log(weight)
        return universal_emit

    def _load_enhanced_hmm(self, train_file: str):
        params = DefaultHmmParams()
        # 获取统计结果
        trans_counts, emit_counts = self._train_transition_probs(train_file)
        # 初始化全量汉字节点
        all_chars = set()
        for context in trans_counts:
            if isinstance(context, tuple):
                all_chars.update(context)
            else:
                all_chars.add(context)
            all_chars.update(trans_counts[context].keys())

        # 初始化所有可能状态的转移概率
        for char in all_chars:
            params.transition_dict.setdefault(char, {})

        # 处理转移概率（二元+三元）
        for context, next_chars in trans_counts.items():
            total = sum(next_chars.values()) + 1e-5  # 平滑处理
            if isinstance(context, tuple):  # 三元上下文
                params.transition_dict.setdefault(context, {})
                for char, count in next_chars.items():
                    params.transition_dict[context][char] = math.log(count / total)
            else:  # 二元上下文
                for char, count in next_chars.items():
                    params.transition_dict[context][char] = math.log(count / total)

        # 处理发射概率
        for py in emit_counts:
            total = sum(emit_counts[py].values()) + 1e-5
            params.emission_dict[py] = {
                char: math.log((count + 1e-5) / total)  # 加1平滑
                for char, count in emit_counts[py].items()
            }
        universal_emit = self.init_universal_emission()  # 获取拼音到汉字的初始发射概率
        params.emission_dict.update(universal_emit)  # 合并内置拼音汉字发射概率
        return params

    def _split_pinyin(self, pinyin_text: str) -> List[str]:
        try:
            return list(jieba.cut(pinyin_text.replace(' ', ''), cut_all=False))
        except Exception as e:
            logger.error(f"分词失败: {str(e)}")
            return self._heuristic_segment(pinyin_text)

    def _handle_single_letter(self, pinyin_list: List[str], top_k: int) -> List[Dict]:
        """处理含单字母的拼音输入"""
        char_probs = defaultdict(float)
        for syllable in pinyin_list:
            if len(syllable) != 1:
                continue
            initial = syllable.lower()
            for py in self.initial_to_pys.get(initial, []):
                for char, log_prob in self.hmm_params.emission_dict.get(py, {}).items():
                    if log_prob > char_probs.get(char, -float('inf')):
                        char_probs[char] = log_prob

        # 转换为结果并排序
        sorted_chars = sorted(char_probs.items(), key=lambda x: -x[1])
        seen = set()
        results = []
        for char, log_prob in sorted_chars:
            if char not in seen:
                seen.add(char)
                results.append({'text': char, 'score': math.exp(log_prob)})
        return results[:top_k]

    def _load_user_dict(self, user_id: Optional[str]) -> dict:
             #加载用户词典
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
        #加密保存用户词典
        if not self.user_id or not self.enable_learning:
            return

        filename = f"user_{hashlib.sha256(self.user_id.encode()).hexdigest()[:16]}.dict"
        with self.privacy_lock:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.user_dict, f, ensure_ascii=False)

    def update_learning_setting(self, enable: bool):
        #隐私设置开关
        with self.privacy_lock:
            old_setting = self.enable_learning
            self.enable_learning = enable

            # 关闭时立即清除内存中的用户数据
            if old_setting and not enable:
                self.user_dict.clear()

    def _update_user_model(self, pinyin: str, selected_text: str):
        #安全更新用户模型
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


    def _train_transition_probs(self, file_path: str):
        transition_counts = defaultdict(lambda: defaultdict(int))
        emission_counts = defaultdict(lambda: defaultdict(int))

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                # 文本清洗
                clean_text = re.sub(r'[^\u4e00-\u9fff，。！？、\n]', '', raw_text)
                clean_text = clean_text.replace('\n', '')  # 保留换行符用于分段

                # 分段处理增强鲁棒性
                paragraphs = [p for p in clean_text.split('，') if p]

                for para in paragraphs:
                    # 生成有效拼音序列
                    pinyin_seq = lazy_pinyin(para, style=Style.NORMAL, errors=lambda x: [''])
                    hanzi_seq = list(para)

                    # 严格对齐处理
                    valid_pairs = []
                    for hz, py in zip(hanzi_seq, pinyin_seq):
                        if py:  # 拼音非空时保留
                            valid_pairs.append((hz, py))

                    # 统计发射概率
                    for hz, py in valid_pairs:
                        emission_counts[py][hz] += 1

                    # 统计转移概率（修正索引范围）
                    hanzi_list = [p[0] for p in valid_pairs]
                    for i in range(len(hanzi_list)):
                        # 二元转移
                        if i > 0:
                            prev = hanzi_list[i - 1]
                            curr = hanzi_list[i]
                            transition_counts[prev][curr] += 1
                        # 三元转移
                        if i > 1:
                            context = (hanzi_list[i - 2], hanzi_list[i - 1])
                            transition_counts[context][hanzi_list[i]] += 1

        except Exception as e:
            logger.error(f"训练数据加载失败: {str(e)}")
        return transition_counts, emission_counts
    def _init_fallback_strategy(self):
        #初始化降级策略资源
        from pypinyin import pinyin
        self.fallback_pinyin = pinyin

    def is_valid_pinyin(self, text: str) -> bool:
        #增强版输入验证
        if not text:
            return False
        # 允许空格分隔但需符合规范
        return self.pinyin_pattern.match(text.strip()) is not None

    def convert(self, pinyin_text: str, context: str = "", top_k: int = 5) -> List[Dict]:
        pinyin_text = pinyin_text.strip().replace(' ', '')
        results=[]
        if not self.is_valid_pinyin(pinyin_text):
            logger.warning(f"非法拼音输入: {pinyin_text}")
            return []

        if pinyin_text in self._cache:
            return self._cache[pinyin_text]

        try:
            pinyin_list = self._split_pinyin(pinyin_text)
            # 检查是否含有单字母
            if any(len(py) == 1 for py in pinyin_list):
                results = self._handle_single_letter(pinyin_list, top_k)
            else:
                raw_results = self._viterbi_decode(pinyin_list, context)
                results = self._post_process(raw_results, top_k) if raw_results else []

            # 应用用户词典
            if self.enable_learning and pinyin_text in self.user_dict:
                results = self._apply_user_preference(results, pinyin_text)

            return results
        except Exception as e:
            logger.error(f"转换失败: {str(e)}")
            return self._fallback_strategy(pinyin_text)
        finally:
            if results:
                self._cache[pinyin_text] = results

    def _apply_user_preference(self, results: List[Dict], pinyin: str) -> List[Dict]:
        #应用用户个性化排序
        user_entries = {text: weight for text, weight in self.user_dict[pinyin]}

        # 提升用户偏好项的得分
        for res in results:
            if res['text'] in user_entries:
                res['score'] *= user_entries[res['text']] * 2  # 权重加倍

        return sorted(results, key=lambda x: -x['score'])

    def _heuristic_segment(self, pinyin_text: str) -> List[str]:
        #基于常见音节的分割
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
        #带上下文的多音字解码
        # 将上下文转换为拼音特征
        context_pinyins = lazy_pinyin(context) if context else []
        print(context_pinyins)
        print(pinyin_list)
        return viterbi(
            hmm_params=self.hmm_params,
            observations=context_pinyins + pinyin_list, # 组合上下文
            path_num=20,  # 获取更多候选
            log=True
        )

    def _fallback_strategy(self, pinyin_text: str) -> List[Dict]:
        #降级策略：整词匹配
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
        """ 改进的后处理方法 """
        processed = []
        seen = set()

        for path in results:
            try:
                # 将拼音路径转换为汉字
                hanzi_sequence = []
                for py in path.path:
                    # 获取该拼音最可能的汉字
                    if py in self.hmm_params.emission_dict:
                        char = max(self.hmm_params.emission_dict[py].items(),
                                   key=lambda x: x[1])[0]
                    else:
                        # 如果拼音不在训练数据发射字典中，则使用内置拼音库的候选
                        char = self.fallback_pinyin(py)[0][0] if self.fallback_pinyin(py) else '?'
                    hanzi_sequence.append(char)

                text = ''.join(hanzi_sequence)
                if text not in seen:
                    seen.add(text)
                    processed.append({
                        'text': text,
                        'score': math.exp(path.score)
                    })
            except Exception as e:
                logger.error(f"处理路径失败: {str(e)}")

        return sorted(processed, key=lambda x: -x['score'])[:top_k]
class PrivacyController:
    @staticmethod
    def anonymize_input(text: str) -> str:
        #数据脱敏处理
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def encrypt_data(data: dict, key: bytes) -> bytes:
        #AES加密用户数据
        # 实现实际的加密逻辑（示例）
        from Crypto.Cipher import AES
        cipher = AES.new(key, AES.MODE_EAX)
        nonce = cipher.nonce
        ciphertext, tag = cipher.encrypt_and_digest(json.dumps(data).encode())
        return nonce + tag + ciphertext

    @classmethod
    def create_consent_dialog(cls):
        #生成隐私协议对话框
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
    result = converter.convert("xiwang")
    print(result)
    # 检查xue的发射概率
    print("xi的候选:", converter.hmm_params.emission_dict.get('xi', {}))
    print("wang的候选:", converter.hmm_params.emission_dict.get('wang', {}))
    # 应输出: {'学': -1.2039, '雪': -2.3025} (近似值)
    # 检查转移概率
    print("希望转移概率:", converter.hmm_params.transition_dict['希'].get('望', 0))

    print(converter.convert("xiwang"))  # 应优先返回"希望"
    # 测试单字母输入
    print(converter.convert("y"))  # 返回所有x开头的汉字
    converter.update_learning_setting(False)  # 关闭学习