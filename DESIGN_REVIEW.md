# MercariClient

```
用户: "1万円以下のヴィンテージデニム見せて"
        ↓ (LLM 理解 + 生成 tool call)   ← 这一步不在 MercariClient 里
{"keyword": "ヴィンテージ デニム", "price_max": 10000}
        ↓ (MercariClient.search)        ← 这一步就是 MercariClient
Mercari 服务器
        ↓ (原始 JSON, 几十个字段)
        ↓ (_to_item 转换 + 裁剪)         ← 这一步也在 MercariClient 里
[Item(id=..., name=..., price=..., condition=...), ...]
        ↓
返回给 agent loop, 再喂给 LLM 做推荐
```

---

## 1. `MercariClient` 的边界

**它管：** 「结构化参数 → 干净数据」这一段。具体是三件事：
- 用给定的参数（keyword / price_min / price_max / condition_at_least / sort）去调用 mercapi，从 Mercari 服务器换回数据
- 把 Mercari 返回的几十个原始字段，裁剪成 `Item` / `ItemDetails` 这两个精简的自有类型（去掉 checksum、pager_id 这类无论什么 query 都用不上的字段）
- 处理这一层的工程问题：请求签名（交给 mercapi）、限流、错误分类、record/replay 缓存、埋点

**它不管：**
- 用户的自然语言怎么变成这组结构化参数——这是 LLM 在 agent loop 里，读了 tool schema 之后自己决定的，`MercariClient` 从头到尾看不到用户原始那句话
- 语义判断——它不判断"这条商品是否符合用户需求"，它只负责把 Mercari 给的东西如实（裁剪后）转达

一句话：它是一台**确定性的数据搬运机器**，给定同样的参数永远产出同样的结果，不含任何"理解"或"判断"的逻辑。所有智能都在 LLM 那一侧。

---

## 2. record/replay 解决的问题

核心目的是**让评测可复现（reproducibility）**，缓存带来的"省时间"是副产品，不是主要目的。

机制：把每次调用的参数（去掉不影响网络请求的字段，比如 `limit`；去掉 `None` 值；排序后）算出一个稳定哈希作为文件名（cassette key）。第一次调用时打真实网络，把规范化后的结果存成文件；之后同样的参数再来，直接读文件，不打网络。

它解决的具体问题：Mercari 上的商品库是活的，一直在变。如果 Day 3 的 eval 每次都打真实 API，你今天跑出来的分数和明天跑出来的分数，差异里混杂着"你改进了 prompt"和"商品库变了"两种因素，根本分不清是谁的功劳。record/replay 把外部世界冻结在某个时间点上，让你能干净地做"改动前 vs 改动后"的 A/B 对比。

局限性（你已经点出来过一次，值得记下）：它只冻结了"外部数据"，冻结不了"LLM 自身的随机性"——同一句用户输入，LLM 两次可能生成字面不同的参数（多个空格），导致 cassette key 不同、缓存未命中。缓解手段是 eval 时用 `temperature=0`，以及规范化参数（排序、去空值）吸收一部分噪音。

---

## 3. 搜索回退行为，对之后设计的影响

**发现：** Mercari 的搜索几乎不会返回"零结果"。哪怕搜索词是纯乱码、根本不存在的东西，它依然会塞几条**长得完全正常但实际不相关**的商品回来——数据结构上分辨不出这是"回退结果"还是"真实匹配"。

**影响一：推翻了一个原定的容错机制。** 我们原本设计"零结果时,agent 自己判断要不要放宽条件重搜"，但既然几乎不会拿到零结果，这个信号永远不会触发。

**影响二：相关性判断的责任必须转移到 LLM 身上，而且要在 system prompt 里明确要求。** 不能假设"返回的东西都是靠谱的"，需要让 LLM 主动做"这批结果整体上是否真的匹配用户需求"的自我核查——如果发现大部分文不对题，应该换关键词重搜或坦白告诉用户没找到，而不是硬凑三个不相关的东西充数交差。

**影响三：Day 3 的 eval 要专门测这个失败模式。** 不能只用"乱码搜索"这一种极端 case 验证，还需要"正常但冷门"的关键词，观察回退行为在真实使用场景下的出现频率，以及 agent 面对这种情况时，是老实说没找到，还是为了完成任务硬凑答案——后者是个真实存在的失败模式，值得单独统计。


# Agent Loop

LLM（不是 Agent）在每一轮（turn）都会看到三样东西：

1. **tools 列表**：每个工具的 name、description、schema，我们写的，全程不变。
2. **system prompt**：我们设计的，全程不变。
3. **累积的 messages**：一条条 append 上去的对话历史，越来越长。每个 turn 会往里加两条——一条 **assistant**（LLM 自己生成的：调了什么工具、往 schema 里填了什么值、说了什么话），紧跟一条 **user**（不是用户打的字，是我们 Python 执行完工具后，把真实结果伪装成 user 角色贴回去的窗口）。这两条严格交替，是下一个 turn 的完整输入。

LLM 看着这三样，每个 turn 生成一批动作（一个或多个 tool_use）：要么调检索类工具（search_mercari / get_item_details）继续收集信息，要么调终止类工具（present_recommendations / ask_clarification）。**注意"终止"本身也是一次工具调用**，所以它同样要先被回填一条 tool_result 的 user 消息，`Agent` 才真正结束这一轮——终止不是跳过流程，是走完流程后停下。`Agent`（Python 代码）只负责执行 LLM 选的工具、把结果回填进 messages，自己不做任何"选择"。

---

## 问题一：一轮 loop 里，messages 被 append 了几次、每次 append 的是什么？

**核心答案：一轮（一个 turn）恰好 append 两次。一次 assistant，一次 user。**

先记住一个总公式：一个 user query 从头到尾，`messages` 的总条数 = **1 + 2 × turn 数**。那个"1"是最开头用户说的那句话，之后每个 turn 都加 2 条。你之前跑 vintage jeans 那次是 4 个 turn，所以是 `1 + 2×4 = 9` 条，跟你亲眼看到的对上了。

**第一次 append —— assistant 消息，装的是 LLM 生成的原始输出**

```python
self.messages.append({"role": "assistant", "content": response.content})
```

`response.content` 是一个 **block 的 list**（block 就是这个 list 里的每一个元素，没别的含义）。这个 list 里可能有：
- `ThinkingBlock`：模型的内心草稿，用户看不到
- `TextBlock`：模型说的话，比如"検索します"
- `ToolUseBlock`：模型的工具调用请求，比如 `name="search_mercari", input={"keyword": "ヴィンテージ ジーンズ"}`

**⚠️ 你之前的误区①**：你说第一次 append 的是"LLM 调用 tool、填的 schema、以及它的输出"——这句话方向对，但要精确成：**append 的是模型选了哪个工具（`name`）、往 schema 里填了什么值（`input`）、以及它同时说的话/想的东西（text/thinking block）**。注意"填的 schema"这个说法不准——schema 是我们定义的固定模具，模型填的是**值**（`input`），不是 schema 本身。

**⚠️ 你之前的误区②（更重要）**：你第一次回答时，**整个漏掉了第二次 append**。这是最关键的遗漏，因为第二次 append 正是问题二的答案所在。

**第二次 append —— user 消息，装的是我们 Python 代码执行工具后拼出来的结果**

```python
self.messages.append({"role": "user", "content": tool_results})
```

**这条 user 消息不是用户打的字，是我们代码伪装成 user 角色塞进去的一个"窗口"，里面贴的是真实执行结果。** 用查天气的例子最好懂：模型上一步生成了 `tool_use: get_weather(city="Tokyo")`——**它只是提了个请求，它自己没查任何天气**。然后我们的 Python 代码真的去调用天气 API，拿到"东京 22°C 晴"，把这个结果拼成：

```python
{"type": "tool_result", "tool_use_id": "<模型给的那个id>", "content": "东京: 22°C, 晴", "is_error": False}
```

这一整步**全程不经过 LLM**。模型是"点菜的人"，我们代码是"跑腿买菜再把菜端回来的人"。放到 Mercari 项目里就是：模型请求 `search_mercari(keyword=...)`，我们代码真的去调 `MercariClient.search()`，把查到的商品列表 `json.dumps` 成字符串，贴进这条 user 消息。

**一句话记牢**：第一次 append = 模型生成的（点菜单）；第二次 append = 我们执行出来的（端上桌的菜）。

---

## 问题二：为什么 present_recommendations 也要回一个 tool_result 才能终止？

**核心答案：因为 API 有一条硬规定——assistant 消息里每一个 tool_use block，都必须在下一条 user 消息里有一个 tool_use_id 对应的 tool_result，一个都不能少。present_recommendations 在 API 眼里也只是一个普通的 tool_use，没有豁免权。**

**⚠️ 你之前的误区**：你一开始把 present_recommendations 当成一个"特殊的、可以直接结束、不用走回填流程"的东西。恰恰相反——**它走的流程跟 search、get_item_details 完全一样，没有任何特殊待遇。** 它唯一的特别之处是"执行起来什么都不做"（回顾 Day 2：它是 schema-only 工具，纯粹为了让最终答案变成结构化对象），但在协议层面，它就是个 tool_use，必须被回一个 tool_result（内容是 `{"status": "delivered"}`）。

**为什么这条规定这么要命，用反例讲最清楚：**

假设我们图省事，看到 present_recommendations 就直接 `return`，不给它配 tool_result。那 `messages` 里最后一条 assistant 消息里，那个 `ToolUseBlock` 就成了"孤儿"——有请求，没回应。

- **单轮对话**：没事。因为对话结束了，这段带孤儿的历史再也不会被发出去。
- **多轮对话**：炸。用户接着问第二句时，`messages` 要作为完整历史整个带上再发给 API，API 一检查发现有个孤儿 tool_use，**直接报 400 拒绝**。而且报错信息完全不会提"多轮"两个字，你会在一个看起来毫不相关的地方查半天，根本想不到是当初 present_recommendations 少回了一个 tool_result 埋的雷。

**所以设计上的处理是**：`agent.py` 里先把这个 turn 的**所有** tool_use block（不管是不是终止类的）全部执行完、全部回填 tool_result，**然后才**判断"这里面有没有终止类工具，有的话结束"。顺序是"先全部回填，再决定停"，不是"看到终止工具立刻停"。这样无论终止工具是单独出现，还是跟一个 search 并排出现在同一个 turn 里，都不会留下孤儿。

---

## 问题三：三层预算分别拦在哪里，为什么光靠 schema 不够？

**⚠️ 你之前的误区**：你只讲出了第三层（loop 的最大循环次数），把前两层漏了，而且说"三层预算不放在 schema 里"——其实**第一层恰恰就在 schema 里**。三层要全讲出来：

**第一层：schema 层（`tools.py` 的 build_tool_schemas）**
在工具定义里写 `maxItems`、参数范围这些约束，比如 get_item_details 的 `item_ids` 标了 `maxItems: 5`。
**作用：引导模型。这是最弱的一层。** 模型看到这个约束，通常会遵守，但**API 不会替我们强制**——模型完全可以无视 schema，一次塞 8 个 id 进来，API 照样放行。所以光靠这层绝对不够，这就是问题问"为什么 schema 不够"的答案。

**第二层：dispatcher 层（`tools.py` 的 ToolExecutor）**
真正生效的硬拦截，比如：
```python
if self.searches_used >= self.budget.max_searches:
    return ToolOutcome(payload=..., is_error=True)
```
**作用：单项强制限制。** 不管模型想不想遵守，代码直接数数、超了就拒绝执行、回一个 is_error 的结果告诉模型"别搜了，用现有的去 present"。这层是"就算模型不听话也拦得住"的那道闸。

**第三层：loop 层（`agent.py` 的 for turn in range(max_turns)）**
就是你唯一讲对的那层——总轮数上限（默认 8）。
**作用：兜底的总量限制。** 万一模型很鸡贼，不停换着不同工具调用来绕过第二层的单项计数（这个搜索到上限了就疯狂调详情、详情也满了就调澄清……），第三层直接用"总共只能转 8 圈"这条硬线砍断，防止无限循环烧钱。

**三层的关系（这句话适合当复习时的一句话总结）**：
> schema 引导模型（弱） → dispatcher 强制单个工具的用量（硬） → loop 强制总轮数（兜底）。

**为什么不能只留第三层？** 因为如果没有第二层的单项限制，模型可能在 8 个 turn 里把预算全砸在 search 一个工具上，get_item_details 一次都没调过，最后第三层砍断时，用户拿到的是一个"搜了一堆但从没看过任何商品详情"的残次结果。**第二层的意义不只是限制总数，是让有限的预算在几个工具之间分配得合理。**


# Eval Data

## generate prompt

```
You are generating a test set for evaluating a Mercari Japan shopping agent's
INTENT-PARSING step only: given a user's natural-language query, the agent
must emit a structured search call. We are testing whether it extracts the
right structured parameters — nothing about ranking or recommendations.

Generate exactly 50 test cases as JSONL (one JSON object per line, no array,
no markdown fences, no commentary before or after).

## The agent's search schema (the ONLY structured fields that exist)

- keyword:        string (free text, Japanese search terms)
- price_min:      integer, JPY, inclusive lower bound        (optional)
- price_max:      integer, JPY, inclusive upper bound        (optional)
- condition_at_least: enum, worst acceptable condition       (optional)
      allowed values, best→worst:
      "new", "like_new", "no_noticeable_damage",
      "slight_damage", "damaged", "poor"

Anything the user says that does NOT map to price or condition (colour, size,
brand, era, model) is NOT a separate field — it belongs inside `keyword`.

## Category quota (fill EXACTLY these counts, 50 total)

1. Japanese normal (8): plain JP query, product + maybe colour/brand/size
   folded into keyword. No price, no condition constraint.
2. English normal (8): plain EN query. The agent must translate intent into
   Japanese keyword(s). No price, no condition constraint.
3. Typo / misspelling (8): mostly ENGLISH typos (e.g. "vintaeg jeens"),
   product still recoverable. Test robustness of intent parsing.
4. Rich-constraint (14), split as:
   - price_max only          × 4
   - price range (min+max)   × 4
   - price_min only          × 4
   - condition only          × 6   (natural language → enum mapping)
   - price + condition       × 8

## Diversity requirements (IMPORTANT)

- Product variety is mandatory. Span MANY categories: electronics, fashion,
  bags, shoes, watches, cosmetics, kitchenware, toys/games, cameras, books,
  sports gear, musical instruments, baby items, furniture, collectibles.
  Do NOT let denim / sneakers / iPhone dominate — no single product type may
  appear more than 3 times across all 50.
- Vary phrasing: some terse ("安いミラーレスカメラ"), some full sentences
  ("子供用のサッカーボールを探しています"), some with brand, some generic.
- Vary the condition wording so the enum-mapping test is real: "美品",
  "新品未使用", "傷なし", "未使用に近い", "状態の良い", "使用感少なめ" etc.
- Do NOT use vague/relative price expressions ("くらい", "前後", "안팎",
  "around", "-ish"). Every price constraint must be a concrete integer.

## Output schema per line

{
  "id": "intent_001",
  "category": "japanese_normal | english_normal | typo | rich_constraint",
  "subtype": "price_max | price_range | price_min | condition | price_condition | null",
  "query": "<the user's raw input, in the stated language>",
  "expected": {
    "keyword_must_include": ["<lemma or brand the keyword MUST contain>"],
    "price_min": <int or null>,
    "price_max": <int or null>,
    "condition_at_least": "<enum value or null>"
  }
}

## Rules for `expected` (this is the grading key — be precise)

- keyword_must_include: list the ESSENTIAL Japanese term(s) a correct search
  must contain (usually the product noun, and brand if the user named one).
  Keep it minimal — do not list optional descriptive words. For English and
  typo cases, put the JAPANESE term the agent should have translated to
  (e.g. query "vintaeg jeens" → keyword_must_include ["デニム"] or ["ジーンズ"]).
- price_* : the exact integer the user stated, else null. Never invent a
  bound the user did not give.
- condition_at_least: map the user's wording to the CLOSEST enum value:
      "新品未使用" → "new"
      "未使用に近い" / "美品" → "like_new"
      "傷なし" / "状態の良い" / "目立った傷なし" → "no_noticeable_damage"
  If the user stated no condition preference, this MUST be null. Do not add a
  condition just because the product is often bought used.
- For normal categories (1,2,3): price_* and condition_at_least are ALL null.

Generate the 50 lines now.
```