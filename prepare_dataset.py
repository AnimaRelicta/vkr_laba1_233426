#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР1, задание 2: разбор HTML-документов из ЕИС (УПД и акты), паспорт данных, дубликаты, утечки, разбиение.

Единица наблюдения: колонка таблицы документа (заголовок + значения из колонки).
Класс = каноническое поле (item_name, quantity, price, ...). Метка ставится правилами по официальному
тексту заголовка, непризнанные заголовки попадают в unmapped_headers.csv для ручной проверки.

Запуск:
    pip install beautifulsoup4
    python prepare_dataset.py --input upd --out out
    python prepare_dataset.py --input upd --out out --split-mode temporal
    python prepare_dataset.py --input upd --out out --group-keys contract,seller

Скрипт только читает HTML как текст, ничего из документов не исполняется.
По умолчанию идентификаторы контрактов, продавцов и покупателей заменяются хешами.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("Нужен beautifulsoup4: pip install beautifulsoup4")

# Порядок правил важен: более специфичные идут раньше общих.
RULES = [
    (r"страна происхождения", "country_of_origin"),
    (r"код товара", "item_code"),
    (r"позици\w* по смете", "estimate_position"),
    (r"п/п|по порядку", "row_number"),
    (r"наименование", "item_name"),
    (r"цена|тариф", "price"),
    (r"единица измерения.*\bкод\b", "unit_code"),
    (r"условное обозначение", "unit_name"),
    (r"единица измерения", "unit"),
    (r"количество|объем", "quantity"),
    (r"акциз", "excise"),
    (r"сумма налога|сумма ндс", "vat_amount"),
    (r"ставка", "vat_rate"),
    (r"с налогом|с ндс", "total_with_vat"),
    (r"без налога|без ндс|стоимость", "amount_without_vat"),
]
# Типичные артефакты вёрстки в заголовках печатных форм ЕИС.
SPLIT_FIXES = {"нало говая": "налоговая"}
HYPHEN = re.compile(r"(?<=[а-яё])-(?=[а-яё]{1,5}\b)")  # «Количе-ство» -> «Количество»
NUM_LABEL = re.compile(r"^(\d{1,2}[а-яa-z]?|[А-ЯA-Z])$", re.I)
SPLITS = ["train", "val", "test"]


def clean(s):
    return re.sub(r"\s+", " ", s.replace("\xa0", " ").replace("­", "")).strip()


def to_int(v, default=1):
    try:
        return max(int(v), 1)
    except (TypeError, ValueError):
        return default


def to_float(s):
    s = s.replace(" ", "").replace(",", ".")
    return float(s) if re.fullmatch(r"-?\d+(\.\d+)?", s) else None


def dmy(s):
    d, m, y = s.split(".")
    return f"{y}-{m}-{d}"


def hid(prefix, value, salt):
    if not value:
        return ""
    return prefix + hashlib.sha1((salt + value).encode()).hexdigest()[:8]


def build_grid(rows):
    """Сетка текстов с раскрытыми colspan и rowspan."""
    grid, spans = [], {}
    for r, tr in enumerate(rows):
        row, c = [], 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while (r, c) in spans:
                row.append(spans.pop((r, c)))
                c += 1
            text = clean(cell.get_text(" ", strip=True))
            cs, rs = to_int(cell.get("colspan")), to_int(cell.get("rowspan"))
            for k in range(cs):
                row.append(text)
                for d in range(1, rs):
                    spans[(r + d, c + k)] = text
            c += cs
        while (r, c) in spans:
            row.append(spans.pop((r, c)))
            c += 1
        grid.append(row)
    width = max((len(r) for r in grid), default=0)
    return [r + [""] * (width - len(r)) for r in grid]


def analyse_grid(grid):
    num_idx = None
    for i, row in enumerate(grid):
        cells = [c for c in row if c]
        if len(cells) >= 5 and sum(bool(NUM_LABEL.match(c)) for c in cells) / len(cells) >= 0.9:
            num_idx = i
            break
    if num_idx is None:
        return None
    header_rows = grid[:num_idx]
    if not any("наименован" in " ".join(r).lower() for r in header_rows):
        return None
    labels = {c: t for c, t in enumerate(grid[num_idx]) if t}
    data, total = [], None
    for row in grid[num_idx + 1:]:
        if any(c.lower().startswith("всего") for c in row):
            total = row
            break
        if any(row[c] for c in labels):
            data.append(row)
    return {"header_rows": header_rows, "labels": labels, "data": data, "total": total}


def find_main_table(soup):
    groups = {}
    for tr in soup.select("tr.main-table"):
        groups.setdefault(id(tr.find_parent("table")), []).append(tr)
    best = None
    for rows in groups.values():
        info = analyse_grid(build_grid(rows))
        if info and (best is None or len(info["data"]) > len(best["data"])):
            best = info
    return best


def norm_header(text):
    t = HYPHEN.sub("", clean(text))
    low = t.lower()
    artifact = t != clean(text)
    for bad, good in SPLIT_FIXES.items():
        if bad in low:
            low, artifact = low.replace(bad, good), True
    return low, artifact


def label_header(path):
    for pat, lab in RULES:
        if re.search(pat, path):
            return lab
    return "unmapped"


def parse_doc(path, args):
    """Возвращает (doc, observations, flags) или (None, [], flags)."""
    raw = path.read_bytes()
    soup = BeautifulSoup(raw.decode("utf-8", errors="replace"), "html.parser")
    title = clean(soup.title.get_text()) if soup.title else ""
    for t in soup(["style", "script"]):
        t.decompose()
    text = clean(soup.get_text(" "))
    stem = path.stem
    main = find_main_table(soup)
    if main is None:
        return None, [], [(stem, "no_main_table", "таблица с номерами колонок не найдена")]

    flags = []
    m = re.search(r"_(\d{8})_(\d{19})$", stem)
    file_date = (f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}") if m else ""
    contract = m.group(2) if m else ""
    if not contract:
        mm = re.search(r"[Рр]еестровый номер[^0-9]{0,80}(\d{19})", text)
        contract = mm.group(1) if mm else ""
    mi = re.search(r"(?:Счет-фактура|Акт о приемке выполненных работ)\s*№\s*(\S+)\s*от\s*(\d{2}\.\d{2}\.\d{4})", text)
    invoice_no, invoice_date = (mi.group(1), dmy(mi.group(2))) if mi else ("", "")
    ms = re.search(r"Подписи заказчика.*?по\s+\d{2}\.\d{2}\.\d{4}\s+(\d{2}\.\d{2}\.\d{4})", text)
    doc_date = dmy(ms.group(1)) if ms else (file_date or invoice_date)
    inn = r"(?<!\d)(\d{12}|\d{10})(?!\d)"
    ms_ = re.search(r"ИНН/КПП продавца\s*" + inn, text) or re.search(r"Подрядчик:.{0,400}?" + inn, text)
    mb_ = re.search(r"ИНН/КПП покупателя\s*" + inn, text) or re.search(r"Заказчик:.{0,300}?ИНН/КПП:\s*" + inn, text)
    seller, buyer = (ms_.group(1) if ms_ else ""), (mb_.group(1) if mb_ else "")
    for name, val in (("contract_id", contract), ("seller_inn", seller), ("buyer_inn", buyer), ("doc_date", doc_date)):
        if not val:
            flags.append((stem, "missing_" + name, ""))

    # Колонки: путь заголовка (верхний уровень / нижний уровень) + значения
    labels, header_rows, data = main["labels"], main["header_rows"], main["data"]
    obs, tokens, seen_labels = [], set(), Counter()
    for c, num_label in labels.items():
        parts = []
        for hr in header_rows:
            t = hr[c]
            if t and (not parts or t != parts[-1]):
                parts.append(t)
        header_raw = " / ".join(parts)
        header_text, artifact = norm_header(header_raw)
        label = label_header(header_text)
        values = [row[c] for row in data]
        filled = [v for v in values if v]
        uniq = list(dict.fromkeys(filled))[:3]
        seen_labels[label] += 1
        if label == "unmapped":
            flags.append((stem, "unmapped_header", header_text))
        if artifact:
            flags.append((stem, "header_artifact", header_raw))
        if label == "price" and any(re.search(r"\.\d{5,}", v) for v in filled):
            flags.append((stem, "price_many_decimals", "цена с длинной дробной частью"))
        tokens.add("H|%s|%s" % (label, header_text))
        for v in filled:
            tokens.add("V|%s|%s" % (label, v.lower()[:200]))
        obs.append({
            "col_label": num_label, "header_raw": header_raw, "header_text": header_text,
            "values_sample": " | ".join(uniq), "n_rows": len(values), "n_values": len(filled),
            "n_numeric": sum(to_float(v) is not None for v in filled), "label": label,
            "header_artifact": int(artifact),
        })
    for lab, n in seen_labels.items():
        if n > 1 and lab not in ("unmapped", "country_of_origin"):
            flags.append((stem, "label_collision", "%s x%d" % (lab, n)))
    if not data:
        flags.append((stem, "no_data_rows", ""))

    # Контроль итога: сумма строк против строки «Всего»
    col_of = {o["label"]: c for (c, _), o in zip(labels.items(), obs)}
    if main["total"] and data:
        for lab in ("amount_without_vat", "total_with_vat"):
            c = col_of.get(lab)
            tv = to_float(main["total"][c]) if c is not None else None
            sv = [to_float(r[c]) for r in data] if c is not None else []
            if tv is not None and sv and None not in sv and abs(sum(sv) - tv) > 0.011:
                flags.append((stem, "total_mismatch", "%s: сумма строк %.2f, итог %.2f" % (lab, sum(sv), tv)))

    template = hashlib.sha1("|".join("%s:%s" % (o["label"], o["header_text"]) for o in obs).encode()).hexdigest()[:8]
    doc = {
        "stem": stem, "file_hash": hashlib.sha1(raw).hexdigest(), "form_type": title or "unknown",
        "template_id": template, "invoice_no": invoice_no, "contract": contract,
        "seller": seller, "buyer": buyer, "doc_date": doc_date, "n_lines": len(data), "tokens": tokens,
    }
    return doc, obs, flags


class DSU:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
            return True
        return False


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def cluster_docs(docs, group_keys, thr):
    """Документы связываются, если у них общий ключ группы, одинаковый файл или Jaccard >= thr."""
    dsu = DSU(list(docs))
    reasons = Counter()
    for key in group_keys:
        first = {}
        for d, doc in docs.items():
            v = doc[key]
            if v:
                if v in first and dsu.union(first[v], d):
                    reasons["общий " + key] += 1
                first.setdefault(v, d)
    first = {}
    for d, doc in docs.items():
        if doc["file_hash"] in first and dsu.union(first[doc["file_hash"]], d):
            reasons["одинаковый файл"] += 1
        first.setdefault(doc["file_hash"], d)
    ids, near_pairs = list(docs), []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if jaccard(docs[ids[i]]["tokens"], docs[ids[j]]["tokens"]) >= thr:
                near_pairs.append((ids[i], ids[j]))
                if dsu.union(ids[i], ids[j]):
                    reasons["почти одинаковое содержимое"] += 1
    groups = defaultdict(list)
    for d in docs:
        groups[dsu.find(d)].append(d)
    return list(groups.values()), reasons, near_pairs


def assign_splits(clusters, docs, mode, ratios, seed):
    total = sum(len(c) for c in clusters)
    assign = {}
    if mode == "temporal":
        acc = 0
        for c in sorted(clusters, key=lambda c: max(docs[d]["doc_date"] or "0000" for d in c)):
            frac = acc / total
            name = SPLITS[0] if frac < ratios[0] else SPLITS[1] if frac < ratios[0] + ratios[1] else SPLITS[2]
            for d in c:
                assign[d] = name
            acc += len(c)
        return assign
    rnd = random.Random(seed)
    cl = clusters[:]
    rnd.shuffle(cl)
    cl.sort(key=len, reverse=True)
    target = {n: r * total for n, r in zip(SPLITS, ratios)}
    cur = {n: 0 for n in SPLITS}
    for c in cl:
        name = max(SPLITS, key=lambda k: target[k] - cur[k])
        for d in c:
            assign[d] = name
        cur[name] += len(c)
    return assign


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def share(a, b):
    return "%.1f %%" % (100 * a / b) if b else "n/a"


def describe(xs):
    if not xs:
        return "n/a"
    return "среднее %.1f, медиана %.1f, мин %d, макс %d" % (statistics.mean(xs), statistics.median(xs), min(xs), max(xs))


def main():
    ap = argparse.ArgumentParser(description="Паспорт данных, дубликаты, утечки и разбиение для ЛР1")
    ap.add_argument("-i", "--input", default="upd", help="папка с .html документами")
    ap.add_argument("-o", "--out", default="out", help="папка для результатов")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="0.7,0.15,0.15", help="доли train,val,test")
    ap.add_argument("--split-mode", choices=["group", "temporal"], default="group")
    ap.add_argument("--group-keys", default="contract", help="ключи групп через запятую: contract, seller, buyer")
    ap.add_argument("--near-dup", type=float, default=0.9, help="порог Jaccard для почти одинаковых документов")
    ap.add_argument("--raw-ids", action="store_true", help="не заменять идентификаторы хешами")
    ap.add_argument("--no-values", action="store_true", help="не писать значения колонок в выходные файлы")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.out)
    if not in_dir.is_dir():
        sys.exit("Папка не найдена: %s" % in_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = [float(x) for x in args.split.split(",")]
    keys = [k.strip() for k in args.group_keys.split(",") if k.strip()]
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-6 or not set(keys) <= {"contract", "seller", "buyer"}:
        sys.exit("Проверьте --split (3 доли, сумма 1) и --group-keys (contract, seller, buyer)")

    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in (".html", ".htm"))
    docs, obs_all, flags = {}, [], []
    for p in files:
        try:
            doc, obs, fl = parse_doc(p, args)
        except Exception as e:  # битый файл не должен ронять весь прогон
            flags.append((p.stem, "parse_failed", repr(e)[:120]))
            continue
        flags += fl
        if doc:
            docs[doc["stem"]] = doc
            for o in obs:
                o["doc"] = doc["stem"]
            obs_all += obs
    if not docs:
        sys.exit("Не разобрано ни одного документа, смотрите quality_flags.csv")

    salt = "lr1"
    rid = (lambda pref, v: v) if args.raw_ids else (lambda pref, v: hid(pref, v, salt))
    clusters, reasons, near_pairs = cluster_docs(docs, keys, args.near_dup)
    split_of = assign_splits(clusters, docs, args.split_mode, ratios, args.seed)
    cluster_id = {d: i for i, c in enumerate(clusters) for d in c}

    for o in obs_all:
        d = docs[o["doc"]]
        o.update({
            "doc_ref": rid("d_", d["stem"]), "split": split_of[d["stem"]], "cluster_id": cluster_id[d["stem"]],
            "form_type": d["form_type"], "template_id": d["template_id"], "doc_date": d["doc_date"],
            "year_month": d["doc_date"][:7], "contract_id": rid("c_", d["contract"]),
            "seller_id": rid("s_", d["seller"]), "buyer_id": rid("b_", d["buyer"]),
        })
        if args.no_values:
            o["values_sample"] = ""
    for i, o in enumerate(obs_all):
        o["obs_id"] = i
    fields = ["obs_id", "split", "doc_ref", "cluster_id", "form_type", "template_id", "col_label", "header_raw",
              "header_text", "values_sample", "n_rows", "n_values", "n_numeric", "label", "header_artifact",
              "contract_id", "seller_id", "buyer_id", "doc_date", "year_month"]
    write_csv(out_dir / "observations.csv", obs_all, fields)
    for s in SPLITS:
        write_csv(out_dir / ("observations_%s.csv" % s), [o for o in obs_all if o["split"] == s], fields)
    write_csv(out_dir / "splits.csv", [
        {"doc_ref": rid("d_", d), "split": split_of[d], "cluster_id": cluster_id[d], "contract_id": rid("c_", docs[d]["contract"]),
         "doc_date": docs[d]["doc_date"], "form_type": docs[d]["form_type"], "template_id": docs[d]["template_id"],
         "n_lines": docs[d]["n_lines"]} for d in docs],
        ["doc_ref", "split", "cluster_id", "contract_id", "doc_date", "form_type", "template_id", "n_lines"])
    write_csv(out_dir / "doc_index_private.csv", [{"doc_ref": rid("d_", d), "file": d} for d in docs], ["doc_ref", "file"])
    write_csv(out_dir / "quality_flags.csv", [{"doc": rid("d_", a), "flag": b, "detail": c} for a, b, c in flags],
              ["doc", "flag", "detail"])
    uniq = Counter((o["label"], o["header_text"]) for o in obs_all)
    write_csv(out_dir / "headers_unique.csv", [{"label": l, "header_text": h, "count": n} for (l, h), n in uniq.most_common()],
              ["label", "header_text", "count"])
    write_csv(out_dir / "unmapped_headers.csv", [{"header_text": h, "count": n} for (l, h), n in uniq.most_common() if l == "unmapped"],
              ["header_text", "count"])

    # ---------- Статистика ----------
    n_docs, n_obs = len(docs), len(obs_all)
    by_label = Counter(o["label"] for o in obs_all)
    lines = [d["n_lines"] for d in docs.values()]
    flag_counts = Counter(f for _, f, _ in flags)
    flag_docs = defaultdict(set)
    for a, b, _ in flags:
        flag_docs[b].add(a)
    miss = defaultdict(lambda: [0, 0, 0])
    for o in obs_all:
        miss[o["label"]][0] += o["n_rows"]
        miss[o["label"]][1] += o["n_rows"] - o["n_values"]
        miss[o["label"]][2] += o["n_numeric"]
    month = Counter(d["doc_date"][:7] or "нет даты" for d in docs.values())
    tmpl_month = defaultdict(Counter)
    for d in docs.values():
        tmpl_month[d["template_id"]][d["doc_date"][:7] or "нет даты"] += 1
    by_split = {s: [d for d in docs if split_of[d] == s] for s in SPLITS}
    split_obs = {s: [o for o in obs_all if o["split"] == s] for s in SPLITS}
    train_headers = {o["header_text"] for o in split_obs["train"]}
    train_pairs = {(o["label"], o["header_text"], o["values_sample"]) for o in split_obs["train"]}

    def overlap(key):
        where = defaultdict(set)
        for d in docs.values():
            if d[key]:
                where[d[key]].add(split_of[d["stem"]])
        return sum(len(v) > 1 for v in where.values()), len(where)

    pair_cross = sum(split_of[a] != split_of[b] for a, b in near_pairs)
    exact_cross = len({(o["label"], o["header_text"], o["values_sample"]) for o in split_obs["val"] + split_obs["test"]} & train_pairs)
    big = max(len(c) for c in clusters)

    P = []
    P.append("# Паспорт данных, ЛР1, задание 2\n")
    P.append("Сформирован %s скриптом prepare_dataset.py, seed %d, режим разбиения: %s, ключи групп: %s, порог почти-дубликатов (Jaccard): %.2f.\n"
             % (datetime.now().strftime("%Y-%m-%d %H:%M"), args.seed, args.split_mode, ", ".join(keys), args.near_dup))
    P.append("## 1. Источник и лицензия\n")
    P.append("Электронные документы о приёмке (УПД, акты) по контрактам 44-ФЗ, выгрузка из ЕИС (zakupki.gov.ru), формат HTML, "
             "архивы вида UPD_<дата>_<реестровый номер контракта>.zip. Лицензия в файлах не указана, условия использования нужно "
             "проверить на сайте источника; в репозиторий и отчёт идут только обезличенные данные (хеши вместо ИНН, контрактов и имён).\n")
    P.append("## 2. Единица наблюдения и классы\n")
    P.append("Единица наблюдения: колонка таблицы документа, то есть заголовок (с учётом двухуровневой шапки) и до трёх значений из колонки. "
             "Класс: каноническое поле. Метка ставится правилами по официальному тексту заголовка; непризнанные заголовки "
             "попадают в unmapped_headers.csv.\n")
    P.append("## 3. Объём\n")
    P.append(md_table(["Показатель", "Значение"], [
        ["HTML-файлов найдено", len(files)], ["разобрано документов", n_docs],
        ["не разобрано", len(files) - n_docs], ["наблюдений (колонок)", n_obs],
        ["уникальных пар (класс, заголовок)", len(uniq)], ["уникальных заголовков", len({h for _, h in uniq})],
        ["шаблонов таблицы (template_id)", len({d["template_id"] for d in docs.values()})],
        ["типов форм", ", ".join(sorted({d["form_type"] for d in docs.values()}))],
        ["строк товаров на документ", describe(lines)],
        ["длина заголовка, символов", describe([len(o["header_text"]) for o in obs_all])]]) + "\n")
    P.append("## 4. Классы и дисбаланс\n")
    P.append(md_table(["Класс", "Наблюдений", "Доля"], [[l, n, share(n, n_obs)] for l, n in by_label.most_common()]))
    nz = [n for n in by_label.values() if n]
    P.append("\nОтношение самого частого класса к самому редкому: %.1f. Все документы одной формы содержат один и тот же набор "
             "колонок, поэтому дисбаланс определяется составом форм, а не числом документов.\n" % (max(nz) / min(nz)))
    P.append("## 5. Пропуски\n")
    P.append(md_table(["Класс", "Ячеек", "Пустых", "Доля пустых", "Числовых"],
                      [[l, v[0], v[1], share(v[1], v[0]), share(v[2], v[0] - v[1])] for l, v in sorted(miss.items())]) + "\n")
    P.append("## 6. Дубликаты и утечка\n")
    P.append(md_table(["Проверка", "Результат"], [
        ["документов с одинаковым файлом", reasons.get("одинаковый файл", 0)],
        ["документов с общим ключом группы", ", ".join("%s: %d" % (k, v) for k, v in reasons.items() if k.startswith("общий")) or "0"],
        ["пар почти одинаковых документов (Jaccard >= %.2f)" % args.near_dup, len(near_pairs)],
        ["кластеров документов (единиц разбиения)", len(clusters)],
        ["размер крупнейшего кластера", "%d из %d (%s)" % (big, n_docs, share(big, n_docs))]]) + "\n")
    P.append("## 7. Временная структура\n")
    P.append("Дата документа: дата подписания заказчиком (приёмки), если её не удалось найти, то дата из имени архива.\n")
    P.append(md_table(["Месяц", "Документов"], sorted(month.items())) + "\n")
    P.append(md_table(["template_id", "Документов", "Месяцы"], [[t, sum(c.values()), ", ".join("%s: %d" % kv for kv in sorted(c.items()))]
                                                              for t, c in tmpl_month.items()]) + "\n")
    P.append("## 8. Проблемы качества\n")
    P.append(md_table(["Флаг", "Случаев", "Документов"], [[f, n, len(flag_docs[f])] for f, n in flag_counts.most_common()]
                      or [["проблем не найдено", 0, 0]]) + "\n")
    P.append("## 9. Разбиение\n")
    P.append(md_table(["Часть", "Документов", "Строк товаров", "Наблюдений"], [
        [s, len(by_split[s]), sum(docs[d]["n_lines"] for d in by_split[s]), len(split_obs[s])] for s in SPLITS]) + "\n")
    P.append("## 10. Проверка утечек после разбиения\n")
    c_over, c_n = overlap("contract")
    s_over, s_n = overlap("seller")
    b_over, b_n = overlap("buyer")
    P.append(md_table(["Проверка", "Результат", "Ожидание"], [
        ["контрактов, попавших в несколько частей", "%d из %d" % (c_over, c_n), "0"],
        ["продавцов в нескольких частях", "%d из %d" % (s_over, s_n), "информационно" if "seller" not in keys else "0"],
        ["покупателей в нескольких частях", "%d из %d" % (b_over, b_n), "информационно" if "buyer" not in keys else "0"],
        ["почти одинаковых пар документов между частями", pair_cross, "0"],
        ["одинаковых (класс, заголовок, значения) в val/test и train", exact_cross, "ожидаемо много: заголовки типовые"],
        ["доля заголовков val, уже встречавшихся в train", share(sum(o["header_text"] in train_headers for o in split_obs["val"]), len(split_obs["val"])), "информационно"],
        ["доля заголовков test, уже встречавшихся в train", share(sum(o["header_text"] in train_headers for o in split_obs["test"]), len(split_obs["test"])), "информационно"]]) + "\n")
    if big > 0.4 * n_docs:
        P.append("Внимание: крупнейший кластер занимает больше 40 %% документов, разбиение получится неравномерным. "
                 "Уберите лишние ключи в --group-keys или ослабьте --near-dup.\n")
    (out_dir / "passport.md").write_text("\n".join(P), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps({
        "docs": n_docs, "files": len(files), "observations": n_obs, "by_label": by_label, "by_month": month,
        "flags": flag_counts, "clusters": len(clusters), "near_pairs": len(near_pairs),
        "split_docs": {s: len(by_split[s]) for s in SPLITS}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(P))
    print("\nГотово: файлы в папке %s" % out_dir)


if __name__ == "__main__":
    main()