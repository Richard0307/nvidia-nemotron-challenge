# Debug Log

---

## 2026-03-26 — CoT 生成 rejected (predicted=None)

### 现象
运行 `generate_cot.py --max-samples 5`，所有样本 rejected，predicted=None。

### 排查过程

**Step 1**：加 model output preview 调试输出
→ 发现 preview 全部显示 `(empty)`，说明模型有返回但 content 为空

**Step 2**：打印原始 API 响应
→ 发现 `{'role': 'assistant', 'content': '', 'refusal': None, 'annotations': []}`
→ content 是空字符串，不是 None，refusal 也是 null

### 根因
`COT_MAX_TOKENS=1200` 对 o4-mini 太小。
o4-mini 是推理模型，`max_completion_tokens` 同时包含**内部推理 token + 输出 token**。
1200 个 token 全被内部推理消耗，输出 content 时已无预算，返回空字符串。

### 修复
`.env` 中将 `COT_MAX_TOKENS` 从 `1200` 改为 `16000`。

---

## 2026-03-26 — CoT 生成 rejected（predicted 含反斜杠）

### 现象
样本3：expected=`'cat imagines book'`，predicted=`'cat\\ imagines\\ book'`，rejected。
模型实际答对了，但被误判为错误。

### 根因
模型在 `\boxed{}` 里使用了 LaTeX 转义空格 `\ `（如 `cat\ imagines\ book`），
`normalize_answer` 未处理该格式，导致字符串比较失败。

### 修复
`normalize_answer` 增加两步清洗：
1. `re.sub(r"\\ ", " ", text)` — 将 `\ ` 替换为普通空格
2. `text.replace("\\", "")` — 移除剩余的反斜杠

### 说明
样本1、2 rejected 是正常的（模型推断出了错误的二进制变换规则），
过滤器行为正确，不需要修复。

---

## 2026-03-29 — Stage 2c notebook 在 `Format and Mix Training Data` 处报 `IndexError`

### 现象
在 `notebooks/nvdia-nemotron-cot-sft.ipynb` 的 `Format and Mix Training Data` 部分执行：

```python
print(cot_texts[0][:700])
```

报错：

```python
IndexError: list index out of range
```

### 根因
表面上是 `cot_texts` 为空，直接访问 `cot_texts[0]` 崩溃；但真正的上游原因在 CoT 清洗阶段：

```python
matches = re.findall(r'\boxed\{([^{}]+)\}', str(text))
```

这里的正则写错了。`r'\boxed...'` 会被正则引擎当成 `\b`（word boundary）处理，而不是匹配字面量 `\boxed{...}`。  
结果是 `extract_last_boxed()` 永远提取不到 boxed 答案，随后 `boxed != answer`，所有 CoT 样本都被过滤掉，最终 `cot_rows = []`，`cot_texts = []`。

### 修复
1. 将 `extract_last_boxed()` 的正则改为匹配字面量反斜杠：

```python
matches = re.findall(r'\\boxed\{([^{}]+)\}', str(text))
```

2. 给预览代码增加空列表保护，避免再次因为调试输出直接崩溃：

```python
print('\n--- Direct sample preview ---')
if direct_texts:
    print(direct_texts[0][:500])
else:
    print('[WARN] direct_texts is empty.')

print('\n--- CoT sample preview ---')
if cot_texts:
    print(cot_texts[0][:700])
else:
    print('[WARN] cot_texts is empty. Check the cleaned CoT count above and verify extract_last_boxed().')
```

### 验证
本地再次运行 `scripts/validate_stage2c_notebook.py` 后结果正常：

- `cot_clean_rows=435`
- `cot_rejected=0`
- `cot_rows_after_oversample=870`
- `mixed_total_rows=5870`

说明这次问题已经修复，Stage 2c notebook 的 CoT 分支重新恢复正常。

---
