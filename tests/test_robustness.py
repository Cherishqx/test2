# -*- coding: utf-8 -*-
"""
RAG.py 鲁棒性测试套件（13 条用例，R01~R13）

测试方法覆盖：有效/无效等价类划分、边界值分析、负面测试（特殊字符/
国际化字符）、判定表驱动、错误推测法、异常场景测试、重复测试（稳定性）。

全部用例不使用 mock，直接驱动真实组件（Qwen 模型 + bge-reranker + ChromaDB）。
"""
import pytest

from rag_test_utils import assert_contract, assert_success, knowledge_texts

pytestmark = pytest.mark.robustness


# ========================================================================
# R01 | 测试方法：有效等价类划分 + 冒烟测试
# 合法中文数学查询属于"有效输入等价类"，应走通完整链路：
# 向量检索 → 重排 → 提示词拼接 → 模型推理 → JSON 组装
# ========================================================================
@pytest.mark.gpu
def test_r01_normal_query_baseline(run_rag):
    entire_json, response, knowledge = run_rag("一元二次方程的定义是什么", history=[])
    parsed = assert_success(entire_json, response)
    texts = knowledge_texts(knowledge)
    assert 1 <= len(texts) <= 5, "正常查询应召回 1~5 条知识（reranker 取 top-5）"
    content = parsed["choices"][0]["message"]["content"]
    assert "方程" in content, "回答内容应与问题主题（方程）相关"


# ========================================================================
# R02 | 测试方法：无效等价类划分（异常输入）
# 空字符串属于"无效输入等价类"，系统应优雅处理：要么正常作答，
# 要么返回受控错误 JSON，绝不能崩溃或返回损坏数据
# ========================================================================
@pytest.mark.gpu
def test_r02_empty_query(run_rag):
    entire_json, response, knowledge = run_rag("", history=[])
    assert_contract(entire_json, response)
    assert isinstance(knowledge, list), "知识列表必须是 list 类型"


# ========================================================================
# R03 | 测试方法：边界值分析（空串与普通串之间的边界状态）
# 纯空白字符（空格/制表符/换行）在语义上等同于空输入，
# 属于有效/无效等价类交界处的边界值
# ========================================================================
@pytest.mark.gpu
def test_r03_whitespace_only_query(run_rag):
    entire_json, response, _ = run_rag(" \t \n\r ", history=[])
    assert_contract(entire_json, response)


# ========================================================================
# R04 | 测试方法：边界值分析（输入长度下边界）
# 单字符 "x" 是最短的非空合法输入，考察极短输入下检索与重排是否稳定
# ========================================================================
@pytest.mark.gpu
def test_r04_single_char_query(run_rag):
    entire_json, response, _ = run_rag("x", history=[])
    assert_contract(entire_json, response)


# ========================================================================
# R05 | 测试方法：边界值分析（输入长度上边界）
# 约 4000 字符的超长合法查询：token 数仍在模型上下文内，
# 验证长文本检索/重排/提示词拼接不溢出、不崩溃
# ========================================================================
@pytest.mark.gpu
def test_r05_long_query_4000_chars(run_rag):
    query = "请详细说明一元二次方程求根公式的推导过程与适用条件。" * 160  # ≈4000 字符
    entire_json, response, _ = run_rag(query, history=[])
    assert_contract(entire_json, response)


# ========================================================================
# R06 | 测试方法：负面测试（特殊字符攻击性输入）
# 键盘符号全集（含引号、反斜杠、反引号等），验证输入不会被拼入
# 提示词/JSON 时引发解析错误
# ========================================================================
@pytest.mark.gpu
def test_r06_special_characters_query(run_rag):
    query = "!@#$%^&*()_+-=[]{}|;:'\",./<>?`~\\·…§¶"
    entire_json, response, _ = run_rag(query, history=[])
    assert_contract(entire_json, response)


# ========================================================================
# R07 | 测试方法：负面测试（国际化/Unicode 字符）
# emoji 与多语言字符（韩文/俄文）属于罕见但合法的 Unicode 输入，
# 验证 embedding、reranker、tokenizer 全链路对 Unicode 的兼容性
# ========================================================================
@pytest.mark.gpu
def test_r07_emoji_unicode_query(run_rag):
    query = "介绍勾股定理 📐✖️➗ sqrt(a²+b²) 🎉 한글 Привет"
    entire_json, response, _ = run_rag(query, history=[])
    assert_contract(entire_json, response)


# ========================================================================
# R08 | 测试方法：等价类划分（英文输入等价类）
# 英文数学问题属于独立的"语言等价类"，验证英文查询也能正常检索作答
# ========================================================================
@pytest.mark.gpu
def test_r08_english_query(run_rag):
    entire_json, response, knowledge = run_rag(
        "How to solve the quadratic equation x^2-5x+6=0?", history=[])
    assert_contract(entire_json, response)
    assert isinstance(knowledge, list)


# ========================================================================
# R09 | 测试方法：等价类划分（混合语言等价类）
# 中英混合 + 半角符号的查询属于"混合语言等价类"，考察分词与检索的兼容性
# ========================================================================
@pytest.mark.gpu
def test_r09_mixed_language_query(run_rag):
    query = "如何用求根公式 quadratic formula 解方程 x^2-5x+6=0？"
    entire_json, response, _ = run_rag(query, history=[])
    assert_contract(entire_json, response)


# ========================================================================
# R10 | 测试方法：等价类划分（领域格式输入等价类）
# LaTeX 公式是数学系统的典型领域输入格式，验证公式符号进入提示词后
# 不破坏模板与 JSON 组装
# ========================================================================
@pytest.mark.gpu
def test_r10_latex_formula_query(run_rag):
    query = (r"求解一元二次方程 $ax^2+bx+c=0$（其中 $a\neq 0$），"
             r"请给出求根公式并说明判别式 $\Delta=b^2-4ac$ 的作用")
    entire_json, response, _ = run_rag(query, history=[])
    assert_contract(entire_json, response)


# ========================================================================
# R11 | 测试方法：判定表驱动测试
# 条件桩：history ∈ {None, 空列表, 非空多轮历史}
# 动作桩：rag() 三种状态下都应正常作答（契约成立）
# 覆盖 RAG.py 中 if history is not None 的两个分支
# ========================================================================
@pytest.mark.gpu
def test_r11_history_state_judgment_table(run_rag):
    query = "一元二次方程的判别式是什么"
    histories = [
        None,
        [],
        [["user", "我想复习一元二次方程"],
         ["assistant", "好的，请问你想了解定义、解法还是判别式？"]],
    ]
    for history in histories:
        entire_json, response, _ = run_rag(query, history=history)
        assert_contract(entire_json, response), f"history={history!r} 时输出违约"


# ========================================================================
# R12 | 测试方法：错误推测法 + 异常场景测试
# 依据经验推测高风险场景：传入不存在的 Chroma 集合名。
# 期望优雅降级（新建空集合 → 空知识 → 仍返回合法 JSON），
# 若本用例失败说明空知识链路存在未捕获异常（缺陷线索）
# ========================================================================
@pytest.mark.gpu
def test_r12_nonexistent_collection(run_rag):
    entire_json, response, knowledge = run_rag(
        "一元二次方程", history=[], collection_name="Not_Exist_Collection_Test")
    parsed = assert_contract(entire_json, response)
    assert isinstance(knowledge, list), "空集合场景知识列表仍须为 list"
    if "choices" in parsed:  # 正常降级路径：知识为空但模型仍作答
        assert isinstance(parsed["choices"][0]["message"]["content"], str)


# ========================================================================
# R13 | 测试方法：重复测试（稳定性/可靠性验证）
# 同一输入连续调用两次：两次输出都满足契约，且检索+重排结果完全一致
# （检索链路确定性），LLM 文本本身不要求逐字一致
# ========================================================================
@pytest.mark.gpu
def test_r13_repeated_call_stability(run_rag):
    query = "勾股定理的内容是什么"
    j1, r1, k1 = run_rag(query, history=[])
    j2, r2, k2 = run_rag(query, history=[])
    assert_contract(j1, r1)
    assert_contract(j2, r2)
    assert knowledge_texts(k1) == knowledge_texts(k2), (
        "同一输入的检索+重排结果应确定一致，否则系统稳定性不足"
    )


# ========================================================================
# R14 | 测试方法：错误推测法（缺陷驱动回归——query=None）
# 缺陷分析：v1 中 chromaRetrieval/Reranker 位于 try 之外，query=None
# 会在检索层抛出未捕获的 TypeError 直接崩溃。
# 优化后（RAG._sanitize_input）：None 被规范化为空查询，走既有空输入
# 路径，输出契约成立。本用例回归守护该优化不被回退
# ========================================================================
@pytest.mark.gpu
def test_r14_none_query_input(run_rag):
    entire_json, response, knowledge = run_rag(None, history=[])
    assert_contract(entire_json, response)
    assert isinstance(knowledge, list), "规范化后知识列表仍须为 list"


# ========================================================================
# R15 | 测试方法：错误推测法（缺陷驱动回归——非字符串 query）
# 缺陷分析：v1 中 query=12345（int）会在检索层抛 TypeError 崩溃。
# 优化后：非字符串输入被安全转为字符串，契约成立
# ========================================================================
@pytest.mark.gpu
def test_r15_non_string_query_input(run_rag):
    entire_json, response, _ = run_rag(12345, history=[])
    assert_contract(entire_json, response)


# ========================================================================
# R16 | 测试方法：边界值分析 + 防御性编程验证（超长历史截断）
# 缺陷分析：v1 对 history 无长度限制，2000 轮历史直接拼入提示词，
# 导致上下文溢出/资源耗尽。
# 优化后（RAG.MAX_HISTORY_TURNS=50）：仅保留最近 50 轮，系统应走
# 成功路径（choices）——本用例断言"成功"而非仅"不崩溃"，以锁定优化效果
# ========================================================================
@pytest.mark.gpu
def test_r16_oversized_history_truncation(run_rag):
    huge_history = []
    for i in range(1000):  # 1000 轮 × 2 条消息 = 2000 条历史
        huge_history.append(["user", f"请再详细讲讲第{i}步配方法的推导细节与系数处理方式"])
        huge_history.append(["assistant", f"好的，第{i}步的关键在于二次项系数的规范化处理。"])
    entire_json, response, _ = run_rag("一元二次方程的解法", history=huge_history)
    parsed = assert_contract(entire_json, response)
    assert "choices" in parsed, "超长历史被截断后应正常作答，而不是溢出失败"
