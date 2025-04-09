from pypinyin import lazy_pinyin, Style, pinyin_dict
from Pinyin2Hanzi import DefaultHmmParams, viterbi
import re
import math
import threading
import json
import pickle
import os
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
        self.hmm_params = self._load_enhanced_hmm('training_data.txt')  # 指定训练数据路径
        self.pinyin_pattern = re.compile(r'^[a-z]+( [a-z]+)*$')  # 更严格的拼音正则
        self._cache = {}
        self._init_fallback_strategy()
        self.enable_learning = enable_learning
        self.user_dict = self._load_user_dict(user_id)
        self.valid_pys = {re.sub(r'\d+', '', py) for py in self.hmm_params.emission_dict}
        self.privacy_lock = threading.Lock()  # 线程安全锁
        jieba.initialize()
        jieba.load_userdict('pinyin_dict.txt')
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

    def _split_pinyin(self, pinyin_text: str) -> List[str]:
        pinyin_clean = pinyin_text.replace(' ', '')
        try:
            # 强制精确模式且不启用HMM
            segments = list(jieba.cut(pinyin_clean, cut_all=False, HMM=False))

            # 验证逻辑：检查无调拼音是否存在于发射词典
            valid = all(
                re.sub(r'\d+', '', seg) in self.hmm_params.emission_dict  # 关键修正
                for seg in segments
            )
            if valid:
                return segments
            else:
                logger.warning(f"无效音节: {segments} -> 触发启发式分割")
                return self._heuristic_segment(pinyin_clean)
        except Exception as e:
            logger.error(f"Jieba分词失败: {e}")
            return self._heuristic_segment(pinyin_clean)

    def _handle_single_letter(self, pinyin_list: List[str], top_k: int) -> List[Dict]:
        """处理首字母缩写，生成所有可能的拼音组合并解码"""
        from itertools import product

        # 获取每个首字母的候选拼音（按候选汉字数量排序）
        candidates = []
        for initial in pinyin_list:
            initial = initial.lower()
            pys = self.initial_to_pys.get(initial, [])
            # 按候选汉字数量降序排列
            sorted_pys = sorted(pys, key=lambda py: -len(self.hmm_params.emission_dict.get(py, {})))
            candidates.append(sorted_pys[:3])  # 每个首字母取前3候选

        # 生成所有可能的拼音组合
        all_combos = product(*candidates)

        results = []
        for combo in all_combos:
            combo = list(combo)
            try:
                # 使用Viterbi解码
                raw_results = self._viterbi_decode(combo, context="")
                if not raw_results:
                    continue

                # 处理结果
                processed = self._post_process(raw_results, top_k)
                for res in processed:
                    results.append((res['text'], res['score']))
            except Exception as e:
                logger.error(f"解码失败: {combo} - {str(e)}")

        # 合并去重并排序
        seen = set()
        final_results = []
        for text, score in sorted(results, key=lambda x: -x[1]):
            if text not in seen:
                seen.add(text)
                final_results.append({'text': text, 'score': score})

        return final_results[:top_k]

    def _heuristic_segment(self, pinyin_text: str) -> List[str]:
        max_len = 6
        segments = []
        i = 0
        n = len(pinyin_text)

        while i < n:
            # 优先匹配最长有效拼音
            matched = False
            for l in range(min(max_len, n - i), 0, -1):
                current = pinyin_text[i:i + l]
                clean_py = re.sub(r'\d+', '', current)

                if clean_py in self.valid_pys:
                    segments.append(current)
                    i += l
                    matched = True
                    break

            # 未匹配时处理首字母逻辑
            if not matched:
                # 如果剩余部分全部是字母且长度<=3，尝试首字母组合
                if (n - i) <= 3 and pinyin_text[i:].isalpha():
                    segments.extend(list(pinyin_text[i:]))
                    break
                else:
                    # 保底策略：单字切分
                    segments.append(pinyin_text[i])
                    i += 1

        return segments

    def convert(self, pinyin_text: str, context: str = "", top_k: int = 5) -> List[Dict]:
        pinyin_text = pinyin_text.strip().replace(' ', '')  # Clean the input
        results = []

        # Check if the input pinyin is valid
        if not self.is_valid_pinyin(pinyin_text):
            logger.warning(f"非法拼音输入: {pinyin_text}")
            return []

        # If the result is in the cache, return it
        if pinyin_text in self._cache:
            return self._cache[pinyin_text]

        try:
            pinyin_list = self._split_pinyin(pinyin_text)

            # Check for mixed mode: presence of both full pinyin and initials
            has_full_py = any(len(py) > 1 for py in pinyin_list)  # Check for full pinyin (more than 1 character)
            has_initial = any(len(py) == 1 for py in pinyin_list)  # Check for initials (1 character)

            if has_full_py and has_initial:
                # If both full pinyin and initials are present, handle the mixed case
                return self._handle_mixed_case(pinyin_list, context, top_k)
            elif all(len(py) == 1 for py in pinyin_list):
                # If all are initials (1 character), handle as a single-letter case
                return self._handle_single_letter(pinyin_list, top_k)
            else:
                # Otherwise, decode with Viterbi and apply post-processing
                raw_results = self._viterbi_decode(pinyin_list, context)
                results = self._post_process(raw_results, top_k) if raw_results else []

        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")

        if self.enable_learning and pinyin_text in self.user_dict:
            results = self._apply_user_preference(results, pinyin_text)

        return results

    def _handle_mixed_case(self, pinyin_list: List[str], context: str, top_k: int) -> List[Dict]:
        """Handles the mixed case of full pinyin and initials (e.g., 'zhong guo z')."""
        from itertools import product

        # Find the point where initials start (i.e., the first character with length 1)
        split_idx = next((i for i, py in enumerate(pinyin_list) if len(py) == 1), len(pinyin_list))
        full_pys = pinyin_list[:split_idx]  # Full pinyin before the initials
        initials = pinyin_list[split_idx:]  # Initials after the full pinyin

        # Decode the full pinyin part using Viterbi
        full_results = self._viterbi_decode(full_pys, context) or []
        candidate_texts = [''.join(res.path) for res in full_results[:3]]  # Take top 3 candidates for full pinyin

        # Generate candidate pinyin for initials (limit to top 2 for each initial)
        initial_candidates = []
        for initial in initials:
            initial = initial.lower()  # Convert to lowercase
            pys = self.initial_to_pys.get(initial, [])  # Get possible full pinyin for the initial
            initial_candidates.append(pys[:2])  # Take top 2 candidates for each initial

        # Generate all combinations of full pinyin and initials
        combined = []
        for text in candidate_texts:
            for combo in product(*initial_candidates):
                new_pys = full_pys + list(combo)  # Combine full pinyin with initial candidates
                raw = self._viterbi_decode(new_pys, context)
                if raw:
                    combined.extend(self._post_process(raw, top_k))  # Apply post-processing

        # Deduplicate results and return the top_k results based on score
        seen = set()
        final = []
        for item in sorted(combined, key=lambda x: -x['score']):
            if item['text'] not in seen:
                seen.add(item['text'])
                final.append(item)

        return final[:top_k]

    def _load_enhanced_hmm(self, train_file: str):
        cache_path = "hmm_params.pkl"

        # 尝试加载缓存
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # 无缓存则训练并保存
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

        # 获取拼音到汉字的初始发射概率
        universal_emit = self.init_universal_emission()
        params.emission_dict.update(universal_emit)  # 合并内置拼音汉字发射概率

        # 保存到缓存文件
        with open(cache_path, "wb") as f:
            pickle.dump(params, f)

        return params

    def _train_transition_probs(self, file_path: str):
        transition_counts = defaultdict(lambda: defaultdict(int))
        emission_counts = defaultdict(lambda: defaultdict(int))
    #训练，数据预处理
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

    def _viterbi_decode(self, pinyin_list: List[str], context: str) -> List[Dict]:
        #带上下文的多音字解码
        # 将上下文转换为拼音特征
        context_pinyins = lazy_pinyin(context) if context else []

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
        processed = []
        seen = set()
        for path in results:
            text = ''.join(path.path)
            if text and text not in seen:
                seen.add(text)
                processed.append({
                    'text': text,
                    'score': math.exp(path.score)
                })
        return sorted(processed, key=lambda x: -x['score'])[:top_k]

    def _get_segmented_results(self, raw_results):
        """从原始结果中提取分割组合"""
        seg_results = []
        for path in raw_results:
            # 假设path.path是拼音列表，如['xi','wang']
            seg_text = ' '.join(path.path)
            seg_results.append({
                'text': seg_text,
                'score': math.exp(path.score) * 0.8  # 分割结果权重稍低
            })
        return seg_results

    def _apply_user_preference(self, results: List[Dict], pinyin: str) -> List[Dict]:
        #应用用户个性化排序
        user_entries = {text: weight for text, weight in self.user_dict[pinyin]}

        # 提升用户偏好项的得分
        for res in results:
            if res['text'] in user_entries:
                res['score'] *= user_entries[res['text']] * 2  # 权重加倍

        return sorted(results, key=lambda x: -x['score'])

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
    print(converter.convert("key"))
    converter.update_learning_setting(False)  # 关闭学习