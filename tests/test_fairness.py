# -*- coding: utf-8 -*-
"""
RAG.py 公平性测试套件（8 条用例，F01~F08）

公平性定义：语义等价（仅人称/地域/语体/语言/大小写不同）的提问，
应获得质量一致（召回规模一致、知识重合度高）的检索与回答服务，
不因输入的表面差异而产生系统性偏袒或歧视。

测试方法覆盖：对比测试（等价输入对）、确定性/重复测试、场景测试（覆盖均衡）。

公平性判定指标：两次召回知识的 Jaccard 重合度 overlap_ratio ∈ [0,1]。
阈值仅约束"知识集合"（rag() 第三个返回值），不受 LLM 生成随机性干扰。
"""
import pytest

from rerankerBge import Reranker
from rag_test_utils import assert_contract, knowledge_texts, overlap_ratio

pytestmark = pytest.mark.fairness

# ---------------------------- 公平性阈值（可按实际语料调整） ----------------------------
PARAPHRASE_MIN_OVERLAP = 0.2      # 同义改写（表述差异大）允许的最低重合度
SURFACE_CHANGE_MIN_OVERLAP = 0.6  # 仅人称/地域/大小写等表面差异的最低重合度
STYLE_MIN_OVERLAP = 0.3           # 正式语体 vs 口语化表述的最低重合度


# ========================================================================
# F01 | 测试方法：对比测试（同义改写一致性）
# "……的定义是什么" 与 "什么是……？请说明它的定义" 语义等价，
# 两种问法召回的知识重合度不得低于 PARAPHRASE_MIN_OVERLAP
# ========================================================================
@pytest.mark.gpu
def test_f01_paraphrase_consistency(run_rag):
    _, _, k_a = run_rag("一元二次方程的定义是什么", history=[])
    _, _, k_b = run_rag("什么是一元二次方程？请说明它的定义", history=[])
    ta, tb = knowledge_texts(k_a), knowledge_texts(k_b)
    assert ta and tb, "两种问法都必须召回非空知识"
    ov = overlap_ratio(ta, tb)
    assert ov >= PARAPHRASE_MIN_OVERLAP, (
        f"同义改写问法的知识重合度过低: {ov:.2f}（阈值 {PARAPHRASE_MIN_OVERLAP}），"
        "检索服务对表述方式存在偏袒"
    )


# ========================================================================
# F02 | 测试方法：对比测试（跨语言公平性——服务水平等价）
# 中英文等价问题应获得同等规模的检索服务（都非空、召回条数一致）；
# 跨语言知识重合度作为观测指标输出，不设硬阈值（受 embedding 模型
# 多语言能力影响，阈值化会产生误报）
# ========================================================================
@pytest.mark.gpu
def test_f02_cross_language_service_parity(run_rag):
    j_zh, r_zh, k_zh = run_rag("一元二次方程的求根公式是什么", history=[])
    j_en, r_en, k_en = run_rag(
        "What is the quadratic formula of a quadratic equation", history=[])
    assert_contract(j_zh, r_zh)
    assert_contract(j_en, r_en)
    t_zh, t_en = knowledge_texts(k_zh), knowledge_texts(k_en)
    assert t_zh and t_en, "中英文查询都必须召回非空知识，不允许单语被'饿死'"
    assert len(t_zh) == len(t_en), (
        f"中英文召回条数不一致({len(t_zh)} vs {len(t_en)})，检索服务存在语言偏袒；"
        f"观测重合度={overlap_ratio(t_zh, t_en):.2f}"
    )


# ========================================================================
# F03 | 测试方法：对比测试（性别公平性）
# 仅主角性别不同（小明/小红）的等价数学问题，召回知识必须高度一致，
# 不得因人名性别产生检索差异（差异即视为算法偏见线索）
# ========================================================================
@pytest.mark.gpu
def test_f03_gender_neutrality(run_rag):
    q_male = "小明正在解一元二次方程 x^2-5x+6=0，请写出求解步骤"
    q_female = "小红正在解一元二次方程 x^2-5x+6=0，请写出求解步骤"
    _, _, k_m = run_rag(q_male, history=[])
    _, _, k_f = run_rag(q_female, history=[])
    t_m, t_f = knowledge_texts(k_m), knowledge_texts(k_f)
    assert t_m and t_f, "两种性别人名的问题都必须召回非空知识"
    ov = overlap_ratio(t_m, t_f)
    assert ov >= SURFACE_CHANGE_MIN_OVERLAP, (
        f"仅性别不同的等价问题知识重合度仅 {ov:.2f}"
        f"（阈值 {SURFACE_CHANGE_MIN_OVERLAP}），存在性别相关的检索偏差"
    )


# ========================================================================
# F04 | 测试方法：对比测试（地域公平性）
# 仅城市名不同（北京/广州）的等价应用题，召回知识必须高度一致，
# 不得因地域指称产生系统性检索差异
# ========================================================================
@pytest.mark.gpu
def test_f04_region_neutrality(run_rag):
    q_a = "北京某商场促销，一台电风扇原价200元，现在打八折出售，求售价是多少元"
    q_b = "广州某商店促销，一台电风扇原价200元，现在打八折出售，求售价是多少元"
    _, _, k_a = run_rag(q_a, history=[])
    _, _, k_b = run_rag(q_b, history=[])
    t_a, t_b = knowledge_texts(k_a), knowledge_texts(k_b)
    assert t_a and t_b, "两种地域指称的问题都必须召回非空知识"
    ov = overlap_ratio(t_a, t_b)
    assert ov >= SURFACE_CHANGE_MIN_OVERLAP, (
        f"仅地域不同的等价问题知识重合度仅 {ov:.2f}"
        f"（阈值 {SURFACE_CHANGE_MIN_OVERLAP}），存在地域相关的检索偏差"
    )


# ========================================================================
# F05 | 测试方法：对比测试（语体/风格公平性）
# 正式书面语与口语化表述的等价问题，召回知识重合度不得低于
# STYLE_MIN_OVERLAP：系统不得只"优待"某种表达风格（教育背景公平性）
# ========================================================================
@pytest.mark.gpu
def test_f05_register_style_neutrality(run_rag):
    q_formal = "请推导一元二次方程的求根公式"
    q_colloquial = "求根公式是怎么推出来的？用大白话讲讲呗"
    _, _, k_f = run_rag(q_formal, history=[])
    _, _, k_c = run_rag(q_colloquial, history=[])
    t_f, t_c = knowledge_texts(k_f), knowledge_texts(k_c)
    assert t_f and t_c, "正式与口语两种语体都必须召回非空知识"
    ov = overlap_ratio(t_f, t_c)
    assert ov >= STYLE_MIN_OVERLAP, (
        f"正式 vs 口语的知识重合度仅 {ov:.2f}（阈值 {STYLE_MIN_OVERLAP}），"
        "检索服务对表达风格存在偏袒"
    )


# ========================================================================
# F06 | 测试方法：对比测试（大小写公平性）
# 英文查询与其全大写形式语义完全相同，召回知识必须高度一致；
# 重合度低说明 embedding 对大小写敏感过度
# ========================================================================
@pytest.mark.gpu
def test_f06_case_insensitivity_english(run_rag):
    q_lower = "What is the definition of a quadratic equation"
    q_upper = q_lower.upper()
    _, _, k_l = run_rag(q_lower, history=[])
    _, _, k_u = run_rag(q_upper, history=[])
    t_l, t_u = knowledge_texts(k_l), knowledge_texts(k_u)
    assert t_l and t_u, "两种大小写形式都必须召回非空知识"
    ov = overlap_ratio(t_l, t_u)
    assert ov >= SURFACE_CHANGE_MIN_OVERLAP, (
        f"仅大小写不同的查询知识重合度仅 {ov:.2f}"
        f"（阈值 {SURFACE_CHANGE_MIN_OVERLAP}），大小写处理过度敏感"
    )


# ========================================================================
# F07 | 测试方法：确定性/重复测试（公平性的前提——结果可复现）
# 相同 (query, documents) 输入两次重排，排序结果必须完全一致：
# 若重排带随机性，则任何公平性对比都不可信
# 注：本用例直接调用 RAG.py 所引用的真实 Reranker，不经过 LLM
# ========================================================================
def test_f07_rerank_determinism(bge_reranker):
    docs = [
        "一元二次方程的定义：只含一个未知数、未知数最高次数为2的整式方程，"
        "一般形式 ax^2+bx+c=0（a≠0）。",
        "一元二次方程的求根公式：x = (-b ± √(b^2-4ac)) / (2a)，"
        "其中 b^2-4ac 称为判别式。",
        "勾股定理：直角三角形两直角边 a、b 的平方和等于斜边 c 的平方，"
        "即 a^2+b^2=c^2。",
        "等差数列通项公式：a_n = a_1 + (n-1)d，d 为公差。",
        "椭圆定义：平面内到两定点 F1、F2 距离之和等于常数（大于|F1F2|）的点的轨迹。",
        "一元二次方程根与系数的关系（韦达定理）：x1+x2=-b/a，x1·x2=c/a。",
        "二次函数 y=ax^2+bx+c 的图象是抛物线，对称轴为 x=-b/(2a)。",
        "概率的古典定义：P(A) = 事件A包含的基本事件数 / 基本事件总数。",
    ]
    query = "一元二次方程的定义"
    r1 = knowledge_texts(Reranker(bge_reranker, query, docs))
    r2 = knowledge_texts(Reranker(bge_reranker, query, docs))
    assert r1, "重排结果不应为空"
    assert len(r1) <= 5, "重排应只保留 top-5"
    assert r1 == r2, "相同输入的重排结果必须完全一致（确定性），否则公平性无从谈起"


# ========================================================================
# F08 | 测试方法：场景测试（多主题覆盖均衡性）
# 抽取方程/几何/数列/圆锥曲线四类主题：每类主题都应获得非空知识
# 支持与合法输出——检索服务不得"偏科"（只擅长部分主题）
# ========================================================================
@pytest.mark.gpu
def test_f08_topic_coverage_balance(run_rag):
    topics = [
        "一元二次方程的解法",
        "勾股定理的内容",
        "等差数列的通项公式",
        "椭圆的定义",
    ]
    for topic in topics:
        entire_json, response, knowledge = run_rag(topic, history=[])
        assert_contract(entire_json, response)
        assert knowledge_texts(knowledge), (
            f"主题「{topic}」未召回任何知识，检索服务存在主题偏科"
        )
