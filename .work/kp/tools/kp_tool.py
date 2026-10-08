#!/usr/bin/env python3
"""《高等工程数学》知识点讲解 —— 辅助工具（只用标准库）。

用法（在仓库根目录运行）：
  python3 .work/kp/tools/kp_tool.py check M3.3            检查一个节（含清单覆盖）
  python3 .work/kp/tools/kp_tool.py check M3              检查章导读 + 该章已存在的各节 + 章级习题覆盖
  python3 .work/kp/tools/kp_tool.py check M               检查部分导读
  python3 .work/kp/tools/kp_tool.py check all [--strict]  检查全部已存在的交付物（--strict：【待核】也算错误）
  python3 .work/kp/tools/kp_tool.py status                列出全部 91 个交付物的状态
  python3 .work/kp/tools/kp_tool.py pages M3.3            打印该节需要阅读的页面图像路径与高清重渲染命令
  python3 .work/kp/tools/kp_tool.py link M3.3 N2.4        打印从 M3.3 文件指向 N2.4 文件的标准 Markdown 链接
                                                          （目标也可以是 M3=章导读、M=部分导读、EX:M3.3=习题解答、INDEX=总目录）
  python3 .work/kp/tools/kp_tool.py index                 重新生成根目录《知识点讲解总目录.md》
  python3 .work/kp/tools/kp_tool.py backlinks [--apply]   在习题解答文件头部插入/更新“知识点讲解”回链（默认只预览）

退出码：有错误时为 1，否则为 0。
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KP = os.path.join(ROOT, ".work", "kp")
DATA = json.load(open(os.path.join(KP, "sections.json"), encoding="utf-8"))
META = DATA["meta"]

SECTION_H2 = ["本节导读", "知识结构", "知识点讲解", "方法与题型归纳", "典型例题",
              "易错点辨析", "本节小结", "自测题", "自测题答案", "习题导航"]
CHAPTER_H2 = ["本章定位", "知识地图", "各节概要", "核心主线与方法",
              "本章公式与结论速查", "学习建议与常见难点", "本章习题分布"]
PART_H2 = ["本部分概述", "章节结构与依赖", "学习路线", "与其他部分的联系", "各章导读"]
HEADER_FIELDS = ["**教材位置**", "**对应习题**", "**前置知识**", "**后续应用**", "**导航**"]
INV_TYPES = {"定义", "定理", "推论", "引理", "性质", "例", "正文例", "式", "表", "图", "算法", "注"}
MUST_CITE = {"定义", "定理", "推论", "引理"}

BANNED = [r"（略）", r"\(略\)", r"证明略", r"从略", r"过程略", r"推导略", r"计算略", r"步骤略", r"细节略",
          r"读者自证", r"留给读者", r"留作练习", r"请读者自行", r"此处省略", r"具体略", r"同理可证（略）"]
SOFT = [r"显然", r"易知", r"易证", r"不难", r"同理可得"]
HTML_RE = re.compile(r"<\s*/?\s*(div|span|br|img|details|summary|sup|sub|font|p|table|tr|td|th|center|u|b|i|a|ul|ol|li)\b|<!--", re.I)
KP_RE = re.compile(r"^(#{3,4}) ([MNS]\d+\.\d+)\.K(\d+)　(\S.*)$")
KP_LOOSE = re.compile(r"^#{2,6} .*[MNS]\d+\.\d+\.K\d+")
LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")

# ---------- 索引 ----------
SECTIONS, CHAPTERS, PARTS = {}, {}, {}
ORDER = []
for p in DATA["parts"]:
    PARTS[p["code"]] = p
    for c in p["chapters"]:
        c["_part"] = p
        CHAPTERS[c["id"]] = c
        for s in c["sections"]:
            s["_part"], s["_chapter"] = p, c
            SECTIONS[s["id"]] = s
            ORDER.append(s["id"])
ROOT_INDEX = META["root_index"]


def planned_files():
    files = {ROOT_INDEX}
    for p in DATA["parts"]:
        files.add(p["guide_file"])
        for c in p["chapters"]:
            files.add(c["guide_file"])
            for s in c["sections"]:
                files.add(s["file"])
                for e in s["exercises"]:
                    files.add(e["file"])
    return {os.path.normpath(f) for f in files}


PLANNED = planned_files()


def read(rel):
    path = os.path.join(ROOT, rel)
    return open(path, encoding="utf-8").read() if os.path.isfile(path) else None


def section_text_label(s, from_part):
    """链接文字：同部分写“第x.y节　标题”，跨部分加“第X部分-”前缀。"""
    prefix = "" if s["_part"]["code"] == from_part else s["_part"]["cn"] + "-"
    return f"{prefix}第{s['no']}节　{s['title']}"


def resolve_target(tok):
    """返回 (相对仓库根的路径, 链接文字所需信息)"""
    if tok == "INDEX":
        return ROOT_INDEX, ("index", None)
    if tok.startswith("EX:"):
        s = SECTIONS[tok[3:]]
        e = s["exercises"][0]
        return e["file"], ("ex", e["label"])
    if tok in SECTIONS:
        return SECTIONS[tok]["file"], ("sec", SECTIONS[tok])
    if tok in CHAPTERS:
        return CHAPTERS[tok]["guide_file"], ("ch", CHAPTERS[tok])
    if tok in PARTS:
        return PARTS[tok]["guide_file"], ("part", PARTS[tok])
    raise SystemExit(f"未知目标：{tok}")


def file_of(tok):
    return resolve_target(tok)[0]


def make_link(from_tok, to_tok):
    src = file_of(from_tok)
    dst, (kind, obj) = resolve_target(to_tok)
    rel = os.path.relpath(os.path.join(ROOT, dst), os.path.dirname(os.path.join(ROOT, src)))
    if not rel.startswith("."):
        rel = "./" + rel
    from_part = from_tok[0] if from_tok not in ("INDEX",) else ""
    if kind == "sec":
        text = section_text_label(obj, from_part)
    elif kind == "ch":
        same = obj["_part"]["code"] == from_part and from_tok.startswith(obj["id"])
        text = "本章导读" if same else f"{obj['_part']['cn']}-第{obj['no']}章导读　{obj['name']}"
    elif kind == "part":
        text = f"{obj['cn']}导读"
    elif kind == "ex":
        text = obj
    else:
        text = "知识点讲解总目录"
    return f"[{text}]({rel})"


# ---------- 文本工具 ----------
def strip_math(text):
    """去掉公式，只留正文。行内 $ 不成对的行原样保留（该行另有报错），以免吞掉其后的正文。"""
    t = re.sub(r"\$\$.*?\$\$", lambda m: " ⟦M⟧ " + "\n" * m.group(0).count("\n"), text, flags=re.S)
    out = []
    for line in t.split("\n"):
        if line.replace("\\$", "").count("$") % 2 == 0:
            line = re.sub(r"(?<!\\)\$(?:\\\$|[^$])+?(?<!\\)\$", "⟦m⟧", line)
        out.append(line)
    return "\n".join(out)


def h2_regions(text):
    lines = text.split("\n")
    heads = [(i, l[3:].strip()) for i, l in enumerate(lines) if l.startswith("## ")]
    regions = {}
    for k, (i, name) in enumerate(heads):
        j = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        regions[name] = "\n".join(lines[i + 1:j])
    return [h for _, h in heads], regions


class Report:
    def __init__(self, name):
        self.name, self.errors, self.warns, self.info = name, [], [], []

    def e(self, m):
        self.errors.append(m)

    def w(self, m):
        self.warns.append(m)

    def i(self, m):
        self.info.append(m)

    def show(self):
        status = "通过" if not self.errors else "未通过"
        print(f"\n=== {self.name}：{status}（错误 {len(self.errors)}，警告 {len(self.warns)}）")
        for m in self.errors:
            print("  [错误] " + m)
        for m in self.warns:
            print("  [警告] " + m)
        for m in self.info:
            print("  [信息] " + m)


def check_common(rel, text, rep, strict):
    """所有讲解文件通用的格式检查。"""
    if "```" in text:
        rep.e("出现代码块 ```（讲解文件中不允许代码块）")
    if text.count("$$") % 2:
        rep.e("$$ 总数为奇数，行间公式未闭合")
    for pat in [r"(?<!\\)\\\(", r"(?<!\\)\\\)", r"(?<!\\)\\\[", r"(?<!\\)\\\]"]:
        if re.search(pat, text):
            rep.e(f"使用了 \\( \\) 或 \\[ \\] 公式定界符（只允许 $ 与 $$）：{pat}")
            break
    for bad in [r"\begin{equation", r"\begin{align}", r"\begin{align*}", r"\label{", r"\eqref{", r"\ref{"]:
        if bad in text:
            rep.e(f"使用了不允许的 LaTeX：{bad}")
    in_display = False
    for n, raw in enumerate(text.split("\n"), 1):
        line = re.sub(r"^(>\s?)+", "", raw).strip()
        cnt = line.count("$$")
        if cnt:
            if not (line.startswith("$$") or line.endswith("$$")):
                rep.e(f"第 {n} 行：行间公式 $$ 与正文混在同一行")
            if cnt % 2:
                in_display = not in_display
            rest = re.sub(r"\$\$.*?\$\$", "", line)
            rest = re.sub(r"\$\$.*$|^.*\$\$", "", rest)
        else:
            rest = line
        if in_display and cnt == 0:
            continue
        rest = rest.replace("\\$", "")
        if rest.count("$") % 2:
            rep.e(f"第 {n} 行：行内公式 $ 不成对（行内公式不得跨行）")
        if raw.lstrip().startswith("|"):
            for seg in re.findall(r"(?<!\\)\$(?:\\\$|[^$])+?(?<!\\)\$", raw):
                if "|" in seg:
                    rep.e(f"第 {n} 行：表格单元格内公式含竖线 |，请改用 \\lvert…\\rvert / \\lVert…\\rVert")
                    break
    plain = strip_math(text)
    if HTML_RE.search(plain):
        rep.e("出现 HTML 标签或注释（不允许）")
    for pat in BANNED:
        for m in re.finditer(pat, plain):
            rep.e(f"禁用表述「{m.group(0)}」")
    soft = {}
    # 引用块（以 > 开头）是教材原话的转述，其中的“显然”等不计入
    own = "\n".join(l for l in plain.split("\n") if not l.lstrip().startswith(">"))
    for pat in SOFT:
        c = len(re.findall(pat, own))
        if c:
            soft[pat] = c
    if soft:
        rep.w("弱化表述（须确认紧随其后给出了理由）：" + "，".join(f"{k}×{v}" for k, v in soft.items()))
    punct = re.findall(r"[一-鿿][,;:!?]|[,;:!?][一-鿿]", plain)
    if punct:
        rep.w(f"中文与半角标点相邻 {len(punct)} 处（应使用全角标点），例：{punct[:3]}")
    pend = plain.count("【待核】")
    if pend:
        (rep.e if strict else rep.w)(f"仍有【待核】标记 {pend} 处")
    # 链接
    base = os.path.dirname(os.path.join(ROOT, rel))
    for m in LINK_RE.finditer(text):
        img, _, target = m.groups()
        if img:
            rep.e(f"出现图片 {target}（v1 不配图）")
            continue
        if target.startswith("http://") or target.startswith("https://"):
            rep.w(f"外部链接 {target}")
            continue
        if "#" in target:
            rep.e(f"链接含锚点 # （只允许链接到文件）：{target}")
            continue
        if target.startswith("/"):
            rep.e(f"链接使用了绝对路径：{target}")
            continue
        dest = os.path.normpath(os.path.relpath(os.path.normpath(os.path.join(base, target)), ROOT))
        if os.path.isfile(os.path.join(ROOT, dest)):
            continue
        if dest in PLANNED:
            rep.i(f"前向链接（目标尚未编写）：{dest}")
        else:
            rep.e(f"链接目标不存在且不在规划清单中：{target}")


def check_section(sid, strict=False, quiet=False):
    s = SECTIONS[sid]
    rep = Report(f"{sid} {s['title']}  ({s['file']})")
    text = read(s["file"])
    if text is None:
        rep.e("文件不存在")
        return rep
    lines = text.split("\n")
    first = next((l for l in lines if l.strip()), "")
    if first != "# " + s["h1"]:
        rep.e(f"一级标题应为「# {s['h1']}」，实际为「{first}」")
    a, b = s["read_book_pages"]
    c, d = s["read_pdf_pages"]
    loc = f"**教材位置**：书页 {a}–{b}（PDF {c}–{d}）"
    if loc not in text:
        rep.e(f"头部缺少或写错教材位置，应为「{loc}」")
    head = text.split("\n## ")[0]
    for fld in HEADER_FIELDS:
        if fld not in head:
            rep.e(f"头部缺少字段 {fld}")
    heads, regions = h2_regions(text)
    if heads != SECTION_H2:
        rep.e(f"二级标题必须依次为 {SECTION_H2}，实际为 {heads}")
    check_common(s["file"], text, rep, strict)

    # 知识点
    kp_region = regions.get("知识点讲解", "")
    kps = []
    for n, l in enumerate(lines, 1):
        m = KP_RE.match(l)
        if m:
            kps.append((n, m.group(1), m.group(2), int(m.group(3)), m.group(4)))
        elif KP_LOOSE.match(l):
            rep.e(f"第 {n} 行：知识点标题格式错误（应为「### {sid}.K序号　名称」，ID 与名称之间用全角空格）：{l}")
    if not kps:
        rep.e("没有找到任何知识点标题")
    expected_level = "####" if s["subsections"] else "###"
    for n, lvl, pre, k, name in kps:
        if pre != sid:
            rep.e(f"第 {n} 行：知识点前缀 {pre} 与本节 {sid} 不符")
        if lvl != expected_level:
            rep.e(f"第 {n} 行：本节{'有' if s['subsections'] else '无'}小节，知识点标题应使用 {expected_level}")
        if "$" in name:
            rep.e(f"第 {n} 行：知识点名称中不得含公式")
    nums = [k for *_, k, _ in kps]
    if nums != list(range(1, len(nums) + 1)):
        rep.e(f"知识点序号必须从 K1 起连续：{nums}")
    if 0 < len(kps) < 3:
        rep.w(f"知识点只有 {len(kps)} 个（一般每节不少于 3 个）")
    kp_ids = {f"{pre}.K{k}" for _, _, pre, k, _ in kps}
    # 小节标题
    if s["subsections"]:
        pos = -1
        for sub in s["subsections"]:
            hd = f"### {sub['no']}　{sub['title']}"
            idx = kp_region.find(hd)
            if idx < 0:
                rep.e(f"缺少小节标题「{hd}」（位于“知识点讲解”内）")
            elif idx < pos:
                rep.e(f"小节标题顺序错误：{hd}")
            pos = max(pos, idx)
    # 每个知识点块
    blocks = re.split(r"\n#{3,4} (?=[MNS]\d+\.\d+\.K\d+　)", "\n" + kp_region)[1:]
    for blk in blocks:
        kid = blk.split("　")[0]
        body = re.split(r"\n### \d+\.\d+\.\d+　", blk)[0]
        if "**教材依据**" not in body:
            rep.e(f"{kid}：缺少「**教材依据**」")
        elif "书页" not in body:
            rep.e(f"{kid}：教材依据中未注明书页")
        if "**理解**" not in body and "**要点**" not in body:
            rep.e(f"{kid}：至少需要「**理解**」或「**要点**」之一")
    # 典型例题
    ex_heads = re.findall(r"^### 例题 (\d+)　", regions.get("典型例题", ""), flags=re.M)
    if not ex_heads:
        rep.e("典型例题中没有「### 例题 N　标题」")
    elif len(ex_heads) < 2:
        rep.w("典型例题只有 1 道（一般 2–4 道）")
    elif ex_heads != [str(i) for i in range(1, len(ex_heads) + 1)]:
        rep.e(f"例题编号不连续：{ex_heads}")
    # 自测题
    q = re.findall(r"^\d+\. ", regions.get("自测题", ""), flags=re.M)
    ans = re.findall(r"^\d+\. ", regions.get("自测题答案", ""), flags=re.M)
    if not 3 <= len(q) <= 8:
        rep.e(f"自测题应为 3–8 道，实际 {len(q)}")
    if len(q) != len(ans):
        rep.e(f"自测题 {len(q)} 道与答案 {len(ans)} 条不一致")
    # 习题导航
    nav = regions.get("习题导航", "")
    if not re.search(r"^\|", nav, flags=re.M):
        rep.e("习题导航中没有表格")
    ex_files = {os.path.normpath(e["file"]) for e in s["exercises"]}
    linked = set()
    base = os.path.dirname(os.path.join(ROOT, s["file"]))
    for m in LINK_RE.finditer(nav):
        linked.add(os.path.normpath(os.path.relpath(os.path.normpath(os.path.join(base, m.group(3))), ROOT)))
    if not ex_files & linked:
        rep.e("习题导航中没有链接到本节对应的习题解答文件")
    # 清单
    inv_path = os.path.join(KP, "inventory", f"{sid}.json")
    if not os.path.isfile(inv_path):
        rep.e(f"缺少清单文件 .work/kp/inventory/{sid}.json")
    else:
        try:
            inv = json.load(open(inv_path, encoding="utf-8"))
        except Exception as ex:  # noqa
            rep.e(f"清单 JSON 无法解析：{ex}")
            inv = None
        if inv:
            check_inventory(sid, s, inv, text, kp_ids, nav, rep)
    check_citations(sid, s, text, kp_ids, rep)
    # 篇幅
    chars = len(re.sub(r"\s", "", text))
    pages = s["read_book_pages"][1] - s["read_book_pages"][0] + 1
    per = chars / pages
    rep.i(f"篇幅：{chars} 字符（去空白，含公式源码），教材 {pages} 页，约 {per:.0f} 字符/页；知识点 {len(kps)} 个")
    lo, hi = length_bounds(pages)
    if chars < lo:
        rep.w(f"篇幅偏短：{chars} < 建议下限 {lo}")
    if chars > hi:
        rep.w(f"篇幅偏长：{chars} > 建议上限 {hi}（考虑精简或确认确有必要）")
    return rep


def length_bounds(pages):
    lo = 8000 + 2500 * pages
    hi = 20000 + 6000 * pages
    return lo, hi


def check_inventory(sid, s, inv, text, kp_ids, nav, rep):
    for key in ["id", "read_pdf_pages", "notation", "items", "errata", "exercises"]:
        if key not in inv:
            rep.e(f"清单缺少字段 {key}")
    if inv.get("id") != sid:
        rep.e(f"清单 id={inv.get('id')} 与 {sid} 不符")
    for it in inv.get("items", []):
        t, no = it.get("type"), str(it.get("no", ""))
        if t not in INV_TYPES:
            rep.e(f"清单条目类型非法：{t}")
            continue
        kp = it.get("kp", "")
        if kp:
            if kp not in kp_ids:
                rep.e(f"清单条目 {t} {no} 指向不存在的知识点 {kp}")
        elif not it.get("skip_reason"):
            rep.e(f"清单条目 {t} {no} 既无 kp 也无 skip_reason")
        if t in MUST_CITE and no and f"{t} {no}" not in text:
            rep.e(f"讲解正文未出现「{t} {no}」（教材的定义/定理/推论/引理必须全部覆盖并按原编号引用）")
        if t == "例" and no and f"例 {no}" not in text:
            rep.w(f"讲解正文未出现「例 {no}」")
    nos = set()
    for ex in inv.get("exercises", []):
        no = str(ex.get("no"))
        nos.add(no)
        for k in ex.get("kp", []):
            if k not in kp_ids:
                rep.e(f"习题 {ex.get('set')} 第 {no} 题指向不存在的知识点 {k}")
        if f"第 {no} 题" not in nav:
            rep.e(f"习题导航中缺少「第 {no} 题」")
    if s["_part"]["code"] == "M":
        total = exercise_total(s["exercises"][0]["file"])
        if total:
            miss = [str(i) for i in range(1, total + 1) if str(i) not in nos]
            if miss:
                rep.e(f"清单 exercises 未覆盖 {s['exercises'][0]['label']} 的第 {','.join(miss)} 题（共 {total} 题）")


PART_OF_CN = {"第一部分": "M", "第二部分": "N", "第三部分": "S"}
CITE_RE = re.compile(r"(第[一二三]部分)?(定义|定理|推论|引理|表) ?(\d+\.\d+-\d+)")
FORMULA_RE = re.compile(r"(第[一二三]部分)?式 ?\((\d+\.\d+-\d+[′']?)\)")
TAG_RE = re.compile(r"\\tag\{(\d+\.\d+-\d+[′']?)\}")
EXAMPLE_RE = re.compile(r"(第[一二三]部分)?(?:第 ?)?(\d+\.\d+) ?节例 ?(\d+)")
DOT_STYLE_RE = re.compile(r"(定义|定理|推论|引理|表|式) ?\(?\d+\.\d+\.\d+")
_INV_CACHE = {}


def inventory_items(sec_id):
    """返回该节清单中的 {(类型, 编号)}；清单不存在时返回 None。"""
    if sec_id not in _INV_CACHE:
        p = os.path.join(KP, "inventory", f"{sec_id}.json")
        if not os.path.isfile(p):
            _INV_CACHE[sec_id] = None
        else:
            try:
                inv = json.load(open(p, encoding="utf-8"))
                if not inv.get("items"):  # 空清单视同尚无清单，不据此判定别节引用错误
                    _INV_CACHE[sec_id] = None
                    return None
                _INV_CACHE[sec_id] = {(it.get("type"), str(it.get("no", "")).replace("′", "'"))
                                      for it in inv.get("items", [])}
            except Exception:  # noqa
                _INV_CACHE[sec_id] = None
    return _INV_CACHE[sec_id]


def check_citations(sid, s, text, kp_ids, rep):
    """防编号幻觉：被引用的定义/定理/式/表/例必须出现在对应节的清单中（清单已存在时）。"""
    own = s["_part"]["code"]
    for m in DOT_STYLE_RE.finditer(text):
        rep.e(f"编号写法错误「{m.group(0)}」：教材编号形如 3.3-1（节号-序号），不是 3.3.1")
    cites = []
    for m in CITE_RE.finditer(text):
        cites.append((m.group(1), m.group(2), m.group(3)))
    for m in FORMULA_RE.finditer(text):
        cites.append((m.group(1), "式", m.group(2)))
    for m in TAG_RE.finditer(text):
        cites.append((None, "式", m.group(1)))
    unknown = set()
    for part_cn, typ, no in cites:
        code = PART_OF_CN[part_cn] if part_cn else own
        sec = code + no.split("-")[0]
        if sec not in SECTIONS:
            rep.e(f"引用「{(part_cn or '')}{typ} {no}」指向不存在的节 {sec}")
            continue
        items = inventory_items(sec)
        if items is None:
            unknown.add(sec)
        elif (typ, no.replace("′", "'")) not in items:
            rep.e(f"引用「{(part_cn or '')}{typ} {no}」在 {sec} 的清单中不存在（疑似编号错误；若为教材印刷错误，请在清单中登记更正后的编号）")
    for m in EXAMPLE_RE.finditer(text):
        code = PART_OF_CN[m.group(1)] if m.group(1) else own
        sec = code + m.group(2)
        if sec not in SECTIONS:
            continue
        items = inventory_items(sec)
        if items is None:
            unknown.add(sec)
        elif ("例", m.group(3)) not in items:
            rep.e(f"引用「{m.group(0)}」在 {sec} 的清单中不存在")
    if unknown:
        rep.i("以下节尚无清单，其编号引用暂未核对：" + "、".join(sorted(unknown)))
    # 知识点 ID 引用
    for m in re.finditer(r"\b([MNS]\d+\.\d+)\.K(\d+)\b", text):
        ref, tsec = f"{m.group(1)}.K{m.group(2)}", m.group(1)
        if tsec == sid:
            if ref not in kp_ids:
                rep.e(f"引用了本节不存在的知识点 {ref}")
        elif tsec in SECTIONS:
            other = read(SECTIONS[tsec]["file"])
            if other is not None and ref not in {k for k, _ in kp_list(other)}:
                rep.e(f"引用了 {tsec} 中不存在的知识点 {ref}")
        else:
            rep.e(f"知识点引用 {ref} 指向不存在的节")


def exercise_total(rel):
    t = read(rel) or ""
    m = re.search(r"本题集共\s*(\d+)\s*题", t)
    return int(m.group(1)) if m else 0


def check_chapter(cid, strict=False):
    c = CHAPTERS[cid]
    rep = Report(f"{cid} 第{c['no']}章导读  ({c['guide_file']})")
    text = read(c["guide_file"])
    if text is None:
        rep.e("文件不存在")
    else:
        first = next((l for l in text.split("\n") if l.strip()), "")
        if first != "# " + c["guide_title"]:
            rep.e(f"一级标题应为「# {c['guide_title']}」，实际为「{first}」")
        heads, _ = h2_regions(text)
        if heads != CHAPTER_H2:
            rep.e(f"二级标题必须依次为 {CHAPTER_H2}，实际为 {heads}")
        check_common(c["guide_file"], text, rep, strict)
        for s in c["sections"]:
            fn = os.path.basename(s["file"])
            if f"]({'./' + fn})" not in text and f"]({fn})" not in text:
                rep.e(f"章导读未链接到 {fn}")
    # 第二、三部分：章级习题覆盖
    if "exercise_set" in c:
        total = exercise_total(c["exercise_set"]["file"])
        got, have_all = {}, True
        for s in c["sections"]:
            p = os.path.join(KP, "inventory", f"{s['id']}.json")
            if not os.path.isfile(p):
                have_all = False
                continue
            for ex in json.load(open(p, encoding="utf-8")).get("exercises", []):
                got.setdefault(str(ex.get("no")), []).append(s["id"])
        if have_all and total:
            miss = [str(i) for i in range(1, total + 1) if str(i) not in got]
            if miss:
                rep.e(f"{c['exercise_set']['label']} 的第 {','.join(miss)} 题未分配到任何一节（共 {total} 题）")
            else:
                rep.i(f"{c['exercise_set']['label']} 共 {total} 题，已全部分配到各节")
        elif total:
            rep.i("本章尚有节缺少清单，暂不做章级习题覆盖检查")
    return rep


def check_part(code, strict=False):
    p = PARTS[code]
    rep = Report(f"{code} {p['cn']}导读  ({p['guide_file']})")
    text = read(p["guide_file"])
    if text is None:
        rep.e("文件不存在")
        return rep
    first = next((l for l in text.split("\n") if l.strip()), "")
    if first != "# " + p["guide_title"]:
        rep.e(f"一级标题应为「# {p['guide_title']}」，实际为「{first}」")
    heads, _ = h2_regions(text)
    if heads != PART_H2:
        rep.e(f"二级标题必须依次为 {PART_H2}，实际为 {heads}")
    check_common(p["guide_file"], text, rep, strict)
    for c in p["chapters"]:
        rel = os.path.relpath(c["guide_file"], p["dir"])
        if f"]({rel})" not in text and f"](./{rel})" not in text:
            rep.e(f"部分导读未链接到 {c['guide_file']}")
    return rep


def cmd_check(args):
    strict = "--strict" in args
    targets = [a for a in args if not a.startswith("--")] or ["all"]
    reps = []
    for t in targets:
        if t == "all":
            for code in PARTS:
                if read(PARTS[code]["guide_file"]) is not None:
                    reps.append(check_part(code, strict))
            for cid in CHAPTERS:
                if read(CHAPTERS[cid]["guide_file"]) is not None:
                    reps.append(check_chapter(cid, strict))
            for sid in ORDER:
                if read(SECTIONS[sid]["file"]) is not None:
                    reps.append(check_section(sid, strict))
        elif t in SECTIONS:
            reps.append(check_section(t, strict))
        elif t in CHAPTERS:
            reps.append(check_chapter(t, strict))
            for s in CHAPTERS[t]["sections"]:
                if read(s["file"]) is not None:
                    reps.append(check_section(s["id"], strict))
        elif t in PARTS:
            reps.append(check_part(t, strict))
        else:
            raise SystemExit(f"未知检查对象：{t}")
    for r in reps:
        r.show()
    bad = sum(1 for r in reps if r.errors)
    print(f"\n合计：{len(reps)} 个文件，{bad} 个未通过。")
    return 1 if bad else 0


def cmd_status(args):
    rows, done = [], 0
    for p in DATA["parts"]:
        rows.append((p["code"], p["guide_file"], read(p["guide_file"]) is not None, None))
        for c in p["chapters"]:
            rows.append((c["id"], c["guide_file"], read(c["guide_file"]) is not None, None))
            for s in c["sections"]:
                ok = read(s["file"]) is not None
                inv = os.path.isfile(os.path.join(KP, "inventory", f"{s['id']}.json"))
                rows.append((s["id"], s["file"], ok, inv))
    print(f"{'ID':<6} {'文件存在':<6} {'清单':<4} {'错误':>4} {'警告':>4}  路径")
    for rid, path, ok, inv in rows:
        ne = nw = "-"
        if ok:
            if rid in SECTIONS:
                r = check_section(rid)
            elif rid in CHAPTERS:
                r = check_chapter(rid)
            else:
                r = check_part(rid)
            ne, nw = len(r.errors), len(r.warns)
            done += 1 if not r.errors else 0
        invs = "" if inv is None else ("有" if inv else "无")
        print(f"{rid:<6} {'是' if ok else '否':<8} {invs:<5} {ne!s:>4} {nw!s:>4}  {path}")
    print(f"\n已通过检查：{done}/{len(rows)}（3 个部分导读 + 17 个章导读 + 70 个节讲解；另有根目录总目录由 index 命令生成）")
    return 0


def cmd_pages(args):
    s = SECTIONS[args[0]]
    a, b = s["read_pdf_pages"]
    print(f"{s['id']} {s['title']}：书页 {s['read_book_pages'][0]}–{s['read_book_pages'][1]}，PDF {a}–{b}")
    for p in range(a, b + 1):
        print("  " + os.path.join(META["pages_dir"], META["page_image_pattern"].format(pdf=p)))
    if s["subsections"]:
        print("小节：" + "；".join(f"{x['no']} {x['title']}（书页 {x['book_page']}）" for x in s["subsections"]))
    print("看不清时高清重渲染（300 dpi）：")
    print(f"  mkdir -p /Users/edwinnull/Documents/.work/pages_hi && pdftoppm -f {a} -l {b} -r 300 -png "
          f"\"{os.path.join(ROOT, META['pdf_file'])}\" /Users/edwinnull/Documents/.work/pages_hi/hi")
    for e in s["exercises"]:
        print(f"对应习题：{e['label']} → {e['file']}（{e['scope']}）")
    return 0


def cmd_link(args):
    print(make_link(args[0], args[1]))
    return 0


def kp_list(text):
    return [(f"{m.group(2)}.K{m.group(3)}", m.group(4)) for m in map(KP_RE.match, text.split("\n")) if m]


def cmd_index(args):
    out = ["# 《高等工程数学》知识点讲解总目录", "",
           "> 本文件由 `.work/kp/tools/kp_tool.py index` 自动生成，请勿手工编辑。", ">",
           "> 结构：部分 → 章 → 节。知识点编号规则：`M`＝第一部分 矩阵论，`N`＝第二部分 数值计算方法，`S`＝第三部分 数理统计；"
           "例如 `M3.3.K2` 表示第一部分第 3.3 节的第 2 个知识点。", ">",
           "> 习题解答见 [README](README.md)。", ""]
    total_kp = total_sec = 0
    for p in DATA["parts"]:
        out += [f"## {p['cn']}　{p['name']}", ""]
        if read(p["guide_file"]) is not None:
            out += [f"[{p['cn']}导读]({p['guide_file']})", ""]
        for c in p["chapters"]:
            out += [f"### 第{c['no']}章　{c['name']}", ""]
            if read(c["guide_file"]) is not None:
                out += [f"[本章导读]({c['guide_file']})", ""]
            out += ["| 节 | 知识点 | 对应习题 |", "|---|---|---|"]
            for s in c["sections"]:
                t = read(s["file"])
                ex = "；".join(f"[{e['label']}]({e['file']})" for e in s["exercises"])
                if t is None:
                    out.append(f"| 第{s['no']}节　{s['title']}（待编写） | — | {ex} |")
                    continue
                total_sec += 1
                kps = kp_list(t)
                total_kp += len(kps)
                kp_txt = "；".join(f"`{k}` {n}" for k, n in kps) or "—"
                out.append(f"| [第{s['no']}节　{s['title']}]({s['file']}) | {kp_txt} | {ex} |")
            out.append("")
    out.insert(6, f"> 当前进度：已完成 {total_sec}/70 节，共 {total_kp} 个知识点。")
    out.insert(7, ">")
    open(os.path.join(ROOT, ROOT_INDEX), "w", encoding="utf-8").write("\n".join(out).rstrip() + "\n")
    print(f"已生成 {ROOT_INDEX}：{total_sec} 节，{total_kp} 个知识点")
    return 0


def cmd_backlinks(args):
    apply = "--apply" in args
    by_file = {}
    for sid in ORDER:
        s = SECTIONS[sid]
        if read(s["file"]) is None:
            continue
        for e in s["exercises"]:
            by_file.setdefault(e["file"], []).append(s)
    for ex_file, secs in by_file.items():
        c = secs[0]["_chapter"]
        base = os.path.dirname(os.path.join(ROOT, ex_file))
        parts = [f"[第{s['no']}节　{s['title']}]({'./' + os.path.relpath(os.path.join(ROOT, s['file']), base)})" for s in secs]
        if read(c["guide_file"]) is not None:
            parts.append(f"[本章导读](./{os.path.relpath(os.path.join(ROOT, c['guide_file']), base)})")
        line = "> **知识点讲解**：" + "｜".join(parts)
        text = read(ex_file)
        lines = text.split("\n")
        old = [i for i, l in enumerate(lines) if l.startswith("> **知识点讲解**：")]
        if old:
            if lines[old[0]] == line:
                print(f"未变化：{ex_file}")
                continue
            lines[old[0]] = line
        else:
            h1 = next(i for i, l in enumerate(lines) if l.startswith("# "))
            i = h1 + 1
            while i < len(lines) and (lines[i].strip() == "" or lines[i].startswith(">")):
                i += 1
            j = i
            while j - 1 > h1 and lines[j - 1].strip() == "":
                j -= 1
            lines[j:j] = ["", line]
        print(("已写入：" if apply else "将写入：") + f"{ex_file}\n    {line}")
        if apply:
            open(os.path.join(ROOT, ex_file), "w", encoding="utf-8").write("\n".join(lines))
    if not apply:
        print("\n（预览模式；确认无误后加 --apply 写入）")
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = sys.argv[1], sys.argv[2:]
    fn = {"check": cmd_check, "status": cmd_status, "pages": cmd_pages, "link": cmd_link,
          "index": cmd_index, "backlinks": cmd_backlinks}.get(cmd)
    if not fn:
        print(__doc__)
        return 2
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
