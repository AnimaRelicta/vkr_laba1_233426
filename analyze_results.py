#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Статистика по результатам прогона LLM (zero-shot, а после — и few-shot).

Читает:   <dir>/preds/<split>__<model>__k<K>__<variant>__<reasoning>.jsonl   (по строке на наблюдение)
          <dir>/runs.jsonl                                                  (сводки прогонов; для модели и контроля)
Считает:  macro-F1, accuracy, MCC с 95% CI (bootstrap по кластерам cluster_id, одинаковые пересэмплы для всех
          моделей -> парные сравнения), доли невалидных ответов по типам (с CI Уилсона), F1 по классам,
          задержки p50/p95, токены, стоимость в ₽, попарное Δ macro-F1 против лучшей модели,
          а при наличии прогонов k>0 — сравнение few-shot против zero-shot.
Пишет:    <dir>/analysis/report.md (заголовок + таблицы + текст выводов), summary.csv, per_class.csv

Запуск:   uv run python analyze_results.py --dir llm_out_hard
          uv run python analyze_results.py --dir llm_out_hard --split test --boot 2000
"""
import argparse, csv, json, math, random, re, statistics as st, sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ₽ за 1M токенов (вход, выход), с НДС; каталог proxyapi.ru на 12.10.2026. None -> стоимость не считается.
PRICES = {
    "qwen/qwen3.8-flash":           (20.0, 65.0),
    "google/gemini-3.5-flash-lite": (91.0, 758.0),
    "openai/gpt-5.6-luna":          (60.0, 360.0),
}
NAME_RE = re.compile(r"^(?P<split>[^_]+)__(?P<slug>.+?)__k(?P<k>\d+)__(?P<var>v\d+)__(?P<red>.+)$")


# ------------------------------------------------------------------ загрузка
def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_runs(d):
    p = d / "runs.jsonl"
    runs = {}
    if p.exists():
        for r in load_jsonl(p):          # при повторных запусках берётся последняя запись
            runs[r["run_id"]] = r
    return runs


def load_all(d, split, runs):
    res = {}
    for f in sorted((d / "preds").glob("*.jsonl")):
        m = NAME_RE.match(f.stem)
        if not m or m["split"] != split:
            continue
        rows = load_jsonl(f)
        if not rows:
            continue
        rid = f.stem
        info = runs.get(rid, {})
        res[rid] = dict(run_id=rid, model=info.get("model") or m["slug"].replace("_", "/", 1), k=int(m["k"]),
                        var=m["var"], red=m["red"], rows=rows, info=info)
    return res


# ------------------------------------------------------------------ метрики
def classify_row(r, labels):
    """-> (метка | None, тип невалидности | None)."""
    pred = (r.get("pred") or "").strip()
    if r.get("error"):
        return None, "api_error"
    if pred in labels:
        return pred, None
    if r.get("finish") == "length":
        return None, "truncated"
    if not (r.get("raw") or "").strip():
        return None, "empty"
    return None, "bad_label"


def macro_f1(g, p):
    tp, fp, fn = Counter(), Counter(), Counter()
    for a, b in zip(g, p):
        if a == b:
            tp[a] += 1
        else:
            fn[a] += 1
            if b is not None:
                fp[b] += 1
    fs = []
    for c in set(g):                       # классы, присутствующие в gold выборки
        d = 2 * tp[c] + fp[c] + fn[c]
        fs.append(2 * tp[c] / d if d else 0.0)
    return sum(fs) / len(fs)


def per_class_f1(g, p):
    tp, fp, fn = Counter(), Counter(), Counter()
    for a, b in zip(g, p):
        if a == b:
            tp[a] += 1
        else:
            fn[a] += 1
            if b is not None:
                fp[b] += 1
    return {c: (2 * tp[c] / (2 * tp[c] + fp[c] + fn[c])) for c in set(g)}


def mcc(g, p):
    n = len(g)
    pp = [x if x is not None else "__invalid__" for x in p]
    c = sum(a == b for a, b in zip(g, pp))
    tk, pk = Counter(g), Counter(pp)
    num = c * n - sum(tk[k] * pk[k] for k in tk)
    den = math.sqrt((n * n - sum(v * v for v in pk.values())) * (n * n - sum(v * v for v in tk.values())))
    return num / den if den else 0.0


def acc(g, p):
    return sum(a == b for a, b in zip(g, p)) / len(g)


def pct(v, lo=2.5, hi=97.5):
    v = sorted(v)
    return v[int(len(v) * lo / 100)], v[min(len(v) - 1, int(len(v) * hi / 100))]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def percentile(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))] if v else float("nan")


def make_resamples(clusters, B, seed):
    """Пересэмплы по кластерам (кластер целиком) — учитывает зависимость наблюдений внутри документа."""
    by = defaultdict(list)
    for i, c in enumerate(clusters):
        by[c].append(i)
    keys = list(by)
    rng = random.Random(seed)
    out = []
    for _ in range(B):
        idx = []
        for _ in keys:
            idx += by[keys[rng.randrange(len(keys))]]
        out.append(idx)
    return out


def boot(fn, g, p, rs):
    return pct([fn([g[i] for i in ix], [p[i] for i in ix]) for ix in rs])


def boot_delta(g, p1, p2, rs):
    return pct([macro_f1([g[i] for i in ix], [p2[i] for i in ix]) - macro_f1([g[i] for i in ix], [p1[i] for i in ix]) for ix in rs])


# ------------------------------------------------------------------ анализ
def analyse(runs_data, args):
    """runs_data: {run_id: run}. Выравнивание по общему множеству obs_id."""
    all_labels = {r["gold"] for run in runs_data.values() for r in run["rows"]}
    common = None
    for run in runs_data.values():
        ids = {r["obs_id"] for r in run["rows"]}
        common = ids if common is None else common & ids
    sizes = {len(run["rows"]) for run in runs_data.values()}
    warn = []
    if len(sizes) > 1 or any(len(run["rows"]) != len(common) for run in runs_data.values()):
        warn.append(f"Размеры прогонов различаются ({sorted(sizes)}); сравнение идёт по общим {len(common)} наблюдениям.")
    order = sorted(common)
    # порядок и gold/cluster берём из первого прогона
    first = next(iter(runs_data.values()))
    meta = {r["obs_id"]: r for r in first["rows"]}
    gold = [meta[i]["gold"] for i in order]
    clusters = [str(meta[i].get("cluster_id", i)) for i in order]
    rs = make_resamples(clusters, args.boot, args.seed)
    for run in runs_data.values():
        by = {r["obs_id"]: r for r in run["rows"]}
        rows = [by[i] for i in order]
        cls = [classify_row(r, all_labels) for r in rows]
        run["pred"] = [c[0] for c in cls]
        run["inv_types"] = Counter(c[1] for c in cls if c[1])
        run["rows_al"] = rows
    return all_labels, order, gold, clusters, rs, warn


def summarize(run, gold, rs):
    p = run["pred"]
    n = len(gold)
    inv = sum(run["inv_types"].values())
    lat = [r["latency"] for r in run["rows_al"] if isinstance(r.get("latency"), (int, float)) and r["latency"] > 0]
    tin = sum(r.get("prompt_tokens") or 0 for r in run["rows_al"])
    tout = sum(r.get("completion_tokens") or 0 for r in run["rows_al"])
    pr = PRICES.get(run["model"])
    f1 = macro_f1(gold, p)
    s = dict(model=run["model"], k=run["k"], run_id=run["run_id"], n=n,
             f1=f1, f1_ci=boot(macro_f1, gold, p, rs),
             acc=acc(gold, p), acc_ci=wilson(round(acc(gold, p) * n), n),
             mcc=mcc(gold, p), mcc_ci=boot(mcc, gold, p, rs) if args_global.boot_mcc else None,
             inv=inv / n, inv_k=inv, inv_ci=wilson(inv, n), inv_types=dict(run["inv_types"]),
             p50=percentile(lat, 50), p95=percentile(lat, 95), tin=tin, tout=tout,
             cost=(tin * pr[0] + tout * pr[1]) / 1e6 if pr else None,
             pcf=per_class_f1(gold, p))
    ref = run["info"].get("macro_f1")
    s["check"] = None if ref is None else f1 - ref
    return s


args_global = None
f2 = lambda x: f"{x:.3f}"
ci2 = lambda c: f"[{c[0]:.3f}; {c[1]:.3f}]"


def fmt_cost(c):
    return "—" if c is None else "%.2f" % c


def short(m):
    return m.split("/")[-1]


def build_report(args, S, zero, gold, order, rs, warn, runs_meta, few):
    L = []
    L.append("# Результаты zero-shot: сопоставление колонок УПД с канонической схемой — сравнение LLM\n")
    n = len(order)
    ncl = len({c for c in gold})
    ncluster = len(set(r.get("cluster_id") for r in next(iter(zero.values()))["rows"]))
    L.append(f"**Выборка:** split `{args.split}`, {n} наблюдений, {ncl} классов, {ncluster} кластеров документов. "
             f"Все модели оценены на одних и тех же наблюдениях, одинаковый формат ответа и бюджет. "
             f"95% CI: bootstrap по кластерам ({args.boot} пересэмплов, seed={args.seed}) для F1/MCC; интервал Уилсона для долей.\n")
    for w in warn:
        L.append(f"> ⚠️ {w}\n")
    Z = sorted(S, key=lambda s: -s["f1"])
    L += ["## 1. Сводная таблица (zero-shot)\n",
          "| Модель | macro-F1 [95% CI] | Accuracy | MCC | Невалидных | p50 / p95, с | Токены in / out | Цена вх/вых, ₽/1M | Затраты, ₽ |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in Z:
        pr = PRICES.get(s["model"])
        price_txt = "%s / %s" % pr if pr else "—"
        L.append(f"| {s['model']} | **{f2(s['f1'])}** {ci2(s['f1_ci'])} | {f2(s['acc'])} | {f2(s['mcc'])} | {s['inv']:.1%} | "
                 f"{s['p50']:.1f} / {s['p95']:.1f} | {s['tin']} / {s['tout']} | {price_txt} | "
                 f"{fmt_cost(s['cost'])} |")
    tot = sum(s["cost"] or 0 for s in S)
    L.append(f"\nСуммарные затраты прогона (по токенам из ответов): **{tot:.2f} ₽** (модели без цены не учтены).\n")

    L += ["## 2. Невалидные ответы\n", "| Модель | Невалидных | Доля [95% CI] | api_error | truncated | empty | bad_label |", "|---|---|---|---|---|---|---|"]
    for s in Z:
        t = s["inv_types"]
        L.append(f"| {s['model']} | {s['inv_k']}/{s['n']} | {s['inv']:.1%} [{s['inv_ci'][0]:.1%}; {s['inv_ci'][1]:.1%}] | "
                 f"{t.get('api_error', 0)} | {t.get('truncated', 0)} | {t.get('empty', 0)} | {t.get('bad_label', 0)} |")
    L.append("\n`api_error` — сбой запроса (таймаут, HTTP); `truncated` — `finish=length`, ответ не уместился в `max_tokens`; "
             "`empty` — пустой ответ; `bad_label` — метки нет в схеме. Невалидный ответ считается ошибкой в метриках.\n")

    best = Z[0]
    gi = {s["run_id"]: zero[s["run_id"]] for s in S}
    L += [f"## 3. Парное сравнение с лучшей моделью ({best['model']})\n",
          "| Модель | Δ macro-F1 к лучшей [95% CI] | Различима? |", "|---|---|---|"]
    tied = []
    for s in Z[1:]:
        d = boot_delta(gold, gi[best["run_id"]]["pred"], gi[s["run_id"]]["pred"], rs)
        mean_d = s["f1"] - best["f1"]
        sig = d[1] < 0
        if not sig:
            tied.append(s["model"])
        L.append(f"| {s['model']} | {mean_d:+.3f} {ci2(d)} | {'да (хуже)' if sig else 'нет'} |")

    classes = sorted({c for s in S for c in s["pcf"]})
    L += ["\n## 4. F1 по классам\n", "| Класс | " + " | ".join(short(s["model"]) for s in Z) + " | среднее |", "|---|" + "---|" * (len(Z) + 1)]
    cmean = {}
    for c in classes:
        vals = [s["pcf"].get(c, float("nan")) for s in Z]
        cmean[c] = st.mean(v for v in vals if v == v)
        L.append(f"| {c} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {cmean[c]:.2f} |")

    # --- выводы
    hard = sorted(cmean, key=cmean.get)[:3]
    cheap_ok = [s for s in S if s["cost"] is not None and (s["model"] == best["model"] or s["model"] in tied)]
    val = min(cheap_ok, key=lambda s: s["cost"]) if cheap_ok else None
    fastest = min(S, key=lambda s: s["p50"])
    worst_inv = max(S, key=lambda s: s["inv"])
    trunc = [s for s in S if s["inv_types"].get("truncated", 0) / s["n"] > 0.02]
    apie = [s for s in S if s["inv_types"].get("api_error", 0) > 0]
    L += ["\n## 5. Выводы\n",
          f"- **Лучшая модель:** {best['model']} — macro-F1 {f2(best['f1'])} {ci2(best['f1_ci'])}; худшая — {Z[-1]['model']} ({f2(Z[-1]['f1'])}). "
          f"Диапазон {f2(Z[-1]['f1'])}–{f2(best['f1'])}."]
    L.append(f"- **Различимость:** с лучшей моделью статистически не различимы ещё {len(tied)} из {len(S) - 1} ({', '.join(short(m) for m in tied) or 'нет'}); "
             f"достоверно хуже: {', '.join(short(s['model']) for s in Z[1:] if s['model'] not in tied) or 'нет'}.")
    if val:
        L.append(f"- **Цена/качество:** среди неотличимых от лучшей по F1 самая дешёвая — {val['model']} "
                 f"({val['cost']:.2f} ₽ за прогон; лучшая модель — {best['cost']:.2f} ₽)." if best["cost"] is not None else
                 f"- **Цена/качество:** самая дешёвая среди неотличимых от лучшей — {val['model']} ({val['cost']:.2f} ₽).")
    heavy = max(S, key=lambda s: s["tout"])
    med_out = st.median(s["tout"] for s in S)
    L.append(f"- **Скорость:** быстрее всех {fastest['model']} (p50 {fastest['p50']:.1f} с)."
             + (f" Больше всех выходных токенов у {heavy['model']} ({heavy['tout']} против медианы {int(med_out)}) — "
                f"это reasoning и главный источник затрат." if heavy["tout"] > 3 * med_out else ""))
    L.append(f"- **Невалидные ответы:** максимум у {worst_inv['model']} — {worst_inv['inv']:.1%}."
             + (f" Обрезание по `max_tokens` (truncated >2%): {', '.join(short(s['model']) for s in trunc)} — их F1 занижен, "
                f"это артефакт бюджета, а не качества; перед few-shot нужно поднять `max_tokens`." if trunc else "")
             + (f" Сбои API/таймауты: {', '.join(short(s['model']) for s in apie)} — повторите такие вызовы." if apie else ""))
    L.append(f"- **Трудные классы** (средний F1 по моделям минимален): " + ", ".join(f"{c} ({cmean[c]:.2f})" for c in hard) + ".")
    L.append(f"- **Ограничения:** одна формулировка инструкции и один прогон (чувствительность к промпту не измерена); "
             f"CI широкие при малом числе кластеров ({ncluster}); метки получены правилами, а не вручную; PEFT не проводилась.")

    if few:
        L += ["\n## 6. Few-shot против zero-shot\n", "| Модель | k | F1 zero [CI] | F1 few [CI] | Δ [95% CI] | Невалидных zero → few | Затраты few, ₽ |", "|---|---|---|---|---|---|---|"]
        for (z, f, d) in few:
            L.append(f"| {f['model']} | {f['k']} | {f2(z['f1'])} {ci2(z['f1_ci'])} | {f2(f['f1'])} {ci2(f['f1_ci'])} | "
                     f"{f['f1'] - z['f1']:+.3f} {ci2(d)}{' ✔' if d[0] > 0 else ''} | {z['inv']:.1%} → {f['inv']:.1%} | "
                     f"{fmt_cost(f['cost'])} |")
        L.append("\n✔ — прирост значим (нижняя граница CI Δ > 0).")

    L += ["\n## 7. Журнал параметров\n", "| run_id | модель | max_tokens | temperature | seed | дата | test_sha | script_sha | dropped_params |", "|---|---|---|---|---|---|---|---|---|"]
    for rid, info in runs_meta:
        L.append(f"| {rid} | {info.get('model', '—')} | {info.get('max_tokens', '—')} | {info.get('temperature', '—')} | {info.get('seed', '—')} | "
                 f"{info.get('ts', '—')} | {info.get('test_sha', '—')} | {info.get('script_sha', '—')} | {info.get('dropped_params', '—')} |")
    chk = [(s["model"], s["check"]) for s in S if s["check"] is not None and abs(s["check"]) > 0.005]
    L.append("\n**Контроль:** macro-F1 из preds " + ("совпадает с runs.jsonl (|Δ| ≤ 0.005) для всех прогонов."
             if not chk else "расходится с runs.jsonl у: " + ", ".join(f"{m} ({d:+.3f})" for m, d in chk)
             + " — проверьте определение macro-F1 (набор классов) или состав наблюдений."))
    return "\n".join(L)


def main():
    global args_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="llm_out_hard", help="папка с preds/ и runs.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variant", default="v0")
    ap.add_argument("--reasoning", default="direct")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot-mcc", action="store_true", help="CI для MCC (медленнее)")
    args = ap.parse_args()
    args_global = args
    d = Path(args.dir)
    runs = load_runs(d)
    data = load_all(d, args.split, runs)
    data = {k: v for k, v in data.items() if v["var"] == args.variant and v["red"] == args.reasoning}
    if not data:
        sys.exit(f"В {d / 'preds'} не найдено файлов {args.split}__*__k*__{args.variant}__{args.reasoning}.jsonl")
    all_labels, order, gold, clusters, rs, warn = analyse(data, args)
    sums = {rid: summarize(run, gold, rs) for rid, run in data.items()}
    zero_ids = [rid for rid, r in data.items() if r["k"] == 3]
    if not zero_ids:
        sys.exit("Нет прогонов с k=0 (zero-shot)")
    S = [sums[r] for r in zero_ids]
    zero = {r: data[r] for r in zero_ids}
    few = []
    for rid, r in data.items():
        if r["k"] > 0:
            z = next((sums[x] for x in zero_ids if data[x]["model"] == r["model"]), None)
            if z:
                zp = data[next(x for x in zero_ids if data[x]["model"] == r["model"])]["pred"]
                few.append((z, sums[rid], boot_delta(gold, zp, r["pred"], rs)))
    runs_meta = [(rid, data[rid]["info"]) for rid in sorted(data)]
    rep = build_report(args, S, zero, gold, order, rs, warn, runs_meta, few)
    out = d / "analysis"
    out.mkdir(exist_ok=True)
    (out / "report.md").write_text(rep, encoding="utf-8")
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "k", "n", "macro_f1", "f1_lo", "f1_hi", "accuracy", "mcc", "invalid_rate", "p50", "p95", "tok_in", "tok_out", "cost_rub"])
        for s in sums.values():
            w.writerow([s["model"], s["k"], s["n"], f"{s['f1']:.4f}", f"{s['f1_ci'][0]:.4f}", f"{s['f1_ci'][1]:.4f}", f"{s['acc']:.4f}",
                        f"{s['mcc']:.4f}", f"{s['inv']:.4f}", s["p50"], s["p95"], s["tin"], s["tout"], "" if s["cost"] is None else f"{s['cost']:.2f}"])
    with (out / "per_class.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cl = sorted({c for s in sums.values() for c in s["pcf"]})
        w.writerow(["model", "k"] + cl)
        for s in sums.values():
            w.writerow([s["model"], s["k"]] + [f"{s['pcf'].get(c, float('nan')):.3f}" for c in cl])
    print(rep)
    print(f"\nФайлы: {out / 'report.md'}, summary.csv, per_class.csv")


if __name__ == "__main__":
    main()
