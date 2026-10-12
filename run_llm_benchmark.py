"""Задание 3: сравнение LLM на классификации заголовков колонок УПД.

Этапы (флаг --stage):
  zero    1-й прогон: zero-shot, все модели, один тестовый набор
  few     2-й прогон: few-shot (k релевантных примеров из train для КАЖДОЙ пары)
  best    3-й прогон: выбор лучшей модели по test(zero+few), тюнинг на VAL
          (формулировка промпта x режим рассуждения), финальный прогон на TEST
  report  4-й этап: таблицы в markdown/CSV для вставки в документ

Ключ API берётся из переменной окружения PROXY_API_KEY (в файл не пишем).
Примеры:
  export PROXY_API_KEY=...            # Windows: set PROXY_API_KEY=...
  python run_llm_benchmark.py --list-models
  python run_llm_benchmark.py --stage zero --limit 30          # пробный прогон
  python run_llm_benchmark.py --stage zero
  python run_llm_benchmark.py --stage few --shots 3
  python run_llm_benchmark.py --stage best
  python run_llm_benchmark.py --stage report
  python run_llm_benchmark.py --stage zero --mock              # без сети, проверка пайплайна
"""
import argparse, csv, hashlib, json, math, os, random, re, statistics, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

# ----------------------------------------------------------------- конфигурация
def load_dotenv():
    """Читает .env из текущей папки и из папки скрипта (без внешних библиотек).
    Уже заданные переменные окружения не перезаписываются."""
    for d in (Path.cwd(), Path(__file__).resolve().parent):
        p = d / ".env"
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
        print(f"[.env] загружен {p}")
        return

load_dotenv()
API_BASE =os.environ.get("PROXY_API_BASE", "https://api.proxyapi.ru/v1")
SEED = 42

# MODEL_INFO_ZERO = {   # id: (вход ₽/1M, выход ₽/1M, роль в сравнении)
#     "qwen/qwen3.8-flash":           (20.0, 65.0, "новое поколение Qwen, дёшево"),
#     "deepseek/deepseek-v4-flash":   (20.0, 40.0, "самый дешёвый выход среди сильных; DeepSeek"),
#     "google/gemma-4-31b-it":        (12.63, 46.32, "открытая модель Google, дёшево"),
#     "google/gemini-3.5-flash-lite": (91.0, 758.0, "закрытая Gemini «lite»; дорогой выход"),
#     "gpt-4o-mini":                  (39, 155, стабильный результат за небольшие деньги"),
#     "openai/gpt-5.6-luna":          (60.0, 360.0, "младшая в линейке OpenAI 5.6"),
#     "qwen/qwen3.8-max-0902":        (290.0, 850.0, "верхний ориентир качества (самая дорогая в списке)"),
# }
MODEL_INFO_FEW = {   # id: (вход ₽/1M, выход ₽/1M, роль в сравнении)
    "qwen/qwen3.8-flash":           (20.0, 65.0, "новое поколение Qwen, дёшево"),
    "google/gemini-3.5-flash-lite": (91.0, 758.0, "закрытая Gemini «lite»; дорогой выход"),
    "openai/gpt-5.6-luna":          (60.0, 360.0, "младшая в линейке OpenAI 5.6"),
}

DEFAULT_MODELS = list(MODEL_INFO_FEW)
PRICES = {m: (v[0], v[1]) for m, v in MODEL_INFO_FEW.items()}   # ₽ за 1M токенов

LABELS = {
    "item_code": "код товара/работы/услуги",
    "row_number": "номер строки (№ п/п)",
    "estimate_position": "позиция по смете",
    "item_name": "наименование товара/работы/услуги",
    "unit": "единица измерения (общая колонка)",
    "unit_code": "код единицы измерения (ОКЕИ)",
    "unit_name": "условное обозначение единицы измерения",
    "quantity": "количество (объём)",
    "price": "цена (тариф) за единицу",
    "amount_without_vat": "стоимость без налога",
    "excise": "сумма акциза",
    "vat_rate": "налоговая ставка",
    "vat_amount": "сумма налога (НДС)",
    "total_with_vat": "стоимость с налогом, всего",
    "country_of_origin": "страна происхождения",
    "unmapped": "ни одно из перечисленных полей",
}
INVALID = "__invalid__"

# ----------------------------------------------------------------- промпты (3 формулировки)
def class_block(style):
    if style == "v1":
        return ", ".join(LABELS)
    return "\n".join(f"- {k}: {v}" for k, v in LABELS.items())

def system_prompt(variant, reasoning):
    out = ('Ответ строго в JSON: {"reasoning": "<1-2 предложения>", "label": "<класс>"}'
           if reasoning == "cot" else 'Ответ строго в JSON: {"label": "<класс>"}')
    if variant == "v0":
        return ("Ты размечаешь колонки таблиц из российских закупочных документов (УПД, акты). "
                "По заголовку колонки и примерам значений определи, какому каноническому полю она соответствует.\n"
                f"Допустимые классы:\n{class_block('v0')}\n{out}. Никакого другого текста.")
    if variant == "v1":
        return (f"Task: map a table column to one canonical field. Allowed labels: {class_block('v1')}. "
                f"Use 'unmapped' if none fits. {out.replace('Ответ строго в JSON', 'Reply with JSON only')}")
    if variant == "v2":
        return ("Вы — эксперт по бухгалтерским первичным документам РФ. Вам дают колонку печатной формы документа. "
                "Выберите единственное подходящее поле из списка; если подходящего нет — unmapped.\n"
                f"Список полей:\n{class_block('v2')}\n{out}.")
    raise ValueError(variant)

def fmt_obs(o, variant):
    h, v = o["header_text"], o["values_sample"] or "—"
    if variant == "v2":
        return f"Значения: {v}\nЗаголовок: {h}"
    if variant == "v1":
        return f"header: {h}\nvalues: {v}"
    return f"Заголовок: {h}\nЗначения: {v}"

def user_prompt(o, variant, shots):
    parts = []
    if shots:
        parts.append("Примеры разметки:")
        for s in shots:
            parts.append(fmt_obs(s, variant) + f"\n=> {s['label']}")
        parts.append("Теперь разметь эту колонку:")
    parts.append(fmt_obs(o, variant))
    return "\n\n".join(parts)

# ----------------------------------------------------------------- данные и few-shot
def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["values_sample"] = " | ".join((r.get("values_sample") or "").split(" | ")[:3])
    return rows

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

def grams(o):
    t = (o["header_text"] + " ## " + o["values_sample"]).lower()
    return {t[i:i + 3] for i in range(max(1, len(t) - 2))}

def pick_shots(o, pool, pool_grams, k, seed, hard):
    """k релевантных примеров из train: сходство по char-3gram Jaccard на (заголовок+значения);
    ничьи разрешаются детерминированно через Random(seed:obs_id). Стараемся брать разные классы."""
    rnd = random.Random(f"{seed}:{o['obs_id']}")
    g = grams(o)
    scored = []
    for i, p in enumerate(pool):
        if hard and p["header_text"].lower() == o["header_text"].lower():
            continue
        pg = pool_grams[i]
        scored.append((len(g & pg) / (len(g | pg) or 1), rnd.random(), i))
    scored.sort(reverse=True)
    chosen, seen, seen_text = [], set(), set()
    for _, _, i in scored:                               # 1-й проход: разные классы и тексты
        p = pool[i]
        if p["label"] in seen or (p["header_text"], p["values_sample"]) in seen_text:
            continue
        chosen.append(p); seen.add(p["label"]); seen_text.add((p["header_text"], p["values_sample"]))
        if len(chosen) == k:
            break
    for _, _, i in scored:                               # добор, если классов не хватило
        if len(chosen) == k:
            break
        if pool[i] not in chosen:
            chosen.append(pool[i])
    rnd.shuffle(chosen)                                  # порядок примеров тоже фиксирован seed'ом
    return chosen

# ----------------------------------------------------------------- клиент API
DROPPED = defaultdict(set)      # параметры, которые модель не поддерживает
LOCK = threading.Lock()

def mock_call(model, messages):
    txt = messages[-1]["content"].lower()
    time.sleep(random.random() * 0.01)
    rules = [("страна", "country_of_origin"), ("код товар", "item_code"), ("п/п", "row_number"),
             ("наименование", "item_name"), ("цена", "price"), ("количество", "quantity"),
             ("ставка", "vat_rate"), ("акциз", "excise"), ("сумма налога", "vat_amount"),
             ("с налогом", "total_with_vat"), ("без налога", "amount_without_vat")]
    last = txt.split("теперь разметь эту колонку:")[-1]
    lab = next((v for k, v in rules if k in last), "unmapped")
    if random.random() < 0.1:
        lab = random.choice(list(LABELS))
    return json.dumps({"label": lab}), {"prompt_tokens": 100, "completion_tokens": 10}

def api_call(model, messages, args, reasoning):
    if args.mock:
        return mock_call(model, messages)
    key = os.environ.get("PROXY_API_KEY")
    if not key:
        sys.exit("Нет PROXY_API_KEY в окружении")
    for attempt in range(args.retries + 1):
        body = {"model": model, "messages": messages}
        d = DROPPED[model]
        body["max_tokens" if "max_completion_tokens" not in d else "max_completion_tokens"] = \
            args.max_tokens if reasoning != "cot" else max(args.max_tokens, 400)
        if "temperature" not in d: body["temperature"] = args.temperature
        if "seed" not in d: body["seed"] = args.seed
        if "response_format" not in d: body["response_format"] = {"type": "json_object"}
        if reasoning.startswith("effort_") and "reasoning_effort" not in d:
            body["reasoning_effort"] = reasoning.split("_", 1)[1]
            body["max_tokens" if "max_completion_tokens" not in d else "max_completion_tokens"] = max(args.max_tokens, 2000)
        try:
            r = requests.post(f"{API_BASE}/chat/completions", json=body, timeout=args.timeout,
                              headers={"Authorization": f"Bearer {key}"})
        except requests.RequestException as e:
            time.sleep(2 ** attempt); err = str(e); continue
        if r.status_code == 200:
            j = r.json()
            u = dict(j.get("usage", {})); u["finish_reason"] = j["choices"][0].get("finish_reason", "")
            return j["choices"][0]["message"].get("content") or "", u
        err = r.text[:300]
        if r.status_code == 400:                              # адаптация под ограничения модели
            low = err.lower(); changed = False
            for p in ("temperature", "seed", "response_format", "reasoning_effort"):
                if p in low and p not in DROPPED[model]:
                    with LOCK: DROPPED[model].add(p)
                    changed = True
            if "max_tokens" in low and "max_completion_tokens" not in DROPPED[model]:
                with LOCK: DROPPED[model].add("max_completion_tokens")   # переключаемся на новое имя параметра
                changed = True
            if changed:
                continue
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(err)

def parse_label(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            lab = str(json.loads(m.group(0)).get("label", "")).strip().lower()
            return lab if lab in LABELS else INVALID
        except Exception:
            pass
    m = re.search(r'"label"\s*:\s*"([^"]+)"', text or "")
    if m and m.group(1).strip().lower() in LABELS:
        return m.group(1).strip().lower()
    return INVALID

# ----------------------------------------------------------------- прогон
def run_id(stage_name, model, shots, variant, reasoning, split):
    return f"{split}__{model.replace('/', '_')}__k{shots}__{variant}__{reasoning}"

def run_one(model, rows, pool, args, shots_k, variant, reasoning, split, tag):
    rid = run_id(tag, model, shots_k, variant, reasoning, split)
    out = Path(args.out) / "preds"; out.mkdir(parents=True, exist_ok=True)
    path = out / f"{rid}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); done[r["obs_id"]] = r
    todo = [o for o in rows if o["obs_id"] not in done]
    pool_grams = [grams(p) for p in pool] if shots_k else []
    sysmsg = system_prompt(variant, reasoning)
    t0 = time.time()

    def work(o):
        shots = pick_shots(o, pool, pool_grams, shots_k, args.seed, args.hard) if shots_k else []
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": user_prompt(o, variant, shots)}]
        s = time.time()
        try:
            text, usage = api_call(model, msgs, args, reasoning); err = ""
        except Exception as e:
            text, usage, err = "", {}, str(e)[:200]
        lat = time.time() - s
        return {"obs_id": o["obs_id"], "cluster_id": o.get("cluster_id", ""), "gold": o["label"],
                "pred": parse_label(text), "raw": text[:300], "latency": round(lat, 3), "error": err,
                "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
                "finish": usage.get("finish_reason", ""), "shots": [x["obs_id"] for x in shots]}

    if todo:
        with ThreadPoolExecutor(args.workers) as ex, open(path, "a", encoding="utf-8") as f:
            for r in ex.map(work, todo):
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); done[r["obs_id"]] = r
    wall = time.time() - t0
    preds = [done[o["obs_id"]] for o in rows]
    m = metrics(preds, args.boot, args.seed)
    m["empty_rate"] = sum(not p["raw"].strip() and not p["error"] for p in preds) / len(preds)
    m["truncated_rate"] = sum(p.get("finish") == "length" for p in preds) / len(preds)
    m.update({"run_id": rid, "split": split, "model": model, "shots": shots_k, "variant": variant,
              "reasoning": reasoning, "wall_s_new": round(wall, 1), "n": len(preds),
              "dropped_params": sorted(DROPPED[model])})
    journal(args, m)
    print(f"[{rid}] macroF1={m['macro_f1']:.3f} [{m['macro_f1_lo']:.3f};{m['macro_f1_hi']:.3f}] "
          f"invalid={m['invalid_rate']:.3f} (api_err={m['api_error_rate']:.2f}, empty={m['empty_rate']:.2f}, "
          f"truncated={m['truncated_rate']:.2f}) p50={m['lat_p50']:.2f}s")
    bad = next((p for p in preds if p["pred"] == INVALID), None)
    if bad:
        print(f"    пример сбоя: error={bad['error'][:150]!r} raw={bad['raw'][:150]!r} finish={bad.get('finish')!r}")
    return m

# ----------------------------------------------------------------- метрики
def f1_macro(gold, pred):
    cls = sorted(set(gold)); fs = {}
    for c in cls:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        fs[c] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return sum(fs.values()) / len(fs), fs

def mcc(gold, pred):
    n = len(gold); cg, cp = Counter(gold), Counter(pred)
    c = sum(g == p for g, p in zip(gold, pred))
    num = c * n - sum(cg[k] * cp[k] for k in cg)
    den = math.sqrt((n * n - sum(v * v for v in cp.values())) * (n * n - sum(v * v for v in cg.values())))
    return num / den if den else 0.0

def metrics(preds, n_boot, seed):
    gold = [p["gold"] for p in preds]; pred = [p["pred"] for p in preds]
    mf, per = f1_macro(gold, pred)
    lat = sorted(p["latency"] for p in preds)
    # кластерный bootstrap (по документам/кластерам), чтобы не завышать точность
    groups = defaultdict(list)
    for i, p in enumerate(preds):
        groups[p["cluster_id"] or p["obs_id"]].append(i)
    keys = list(groups); rnd = random.Random(seed); boots = []
    for _ in range(n_boot):
        idx = [i for k in (rnd.choice(keys) for _ in keys) for i in groups[k]]
        boots.append(f1_macro([gold[i] for i in idx], [pred[i] for i in idx])[0])
    boots.sort()
    return {
        "macro_f1": mf, "macro_f1_lo": boots[int(0.025 * len(boots))] if boots else mf,
        "macro_f1_hi": boots[int(0.975 * len(boots)) - 1] if boots else mf,
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold), "mcc": mcc(gold, pred),
        "invalid_rate": sum(p == INVALID for p in pred) / len(pred),
        "api_error_rate": sum(bool(p["error"]) for p in preds) / len(preds),
        "lat_mean": statistics.mean(lat), "lat_p50": lat[len(lat) // 2], "lat_p95": lat[int(0.95 * len(lat)) - 1],
        "prompt_tokens": sum(p["prompt_tokens"] for p in preds), "completion_tokens": sum(p["completion_tokens"] for p in preds),
        "per_class_f1": {k: round(v, 3) for k, v in per.items()},
    }

def journal(args, m):
    p = Path(args.out) / "runs.jsonl"
    rec = dict(m, ts=datetime.now().isoformat(timespec="seconds"), seed=args.seed, temperature=args.temperature,
               max_tokens=args.max_tokens, boot=args.boot, hard_fewshot=args.hard, mock=args.mock,
               test_sha=sha(args.test), train_sha=sha(args.train), val_sha=sha(args.val),
               script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12])
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def read_runs(args):
    p = Path(args.out) / "runs.jsonl"
    if not p.exists():
        return []
    d = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line); d[r["run_id"]] = r
    return list(d.values())

# ----------------------------------------------------------------- этапы
def stage_grid(args, tag, k):
    rows, pool = load(args.test), load(args.train)
    if args.limit:
        rows = random.Random(args.seed).sample(rows, min(args.limit, len(rows)))
    for model in args.models:
        run_one(model, rows, pool, args, k, args.variant, "direct", "test", tag)

def stage_best(args):
    runs = [r for r in read_runs(args) if r["split"] == "test" and r["reasoning"] == "direct" and r["variant"] == args.variant]
    if not runs:
        sys.exit("Сначала выполните --stage zero и --stage few")
    best = max(runs, key=lambda r: (round(r["macro_f1"], 4), -r["lat_p50"]))
    model = args.model or best["model"]
    print(f"Лучшая модель по test: {model} (k={best['shots']}, macro-F1={best['macro_f1']:.3f})")
    tune_k = best["shots"] if not args.model else args.shots
    val, test, pool = load(args.val), load(args.test), load(args.train)
    if args.limit:
        val = random.Random(args.seed).sample(val, min(args.limit, len(val)))
        test = random.Random(args.seed).sample(test, min(args.limit, len(test)))
    modes = args.reasoning_modes.split(",")
    # 1) тюнинг на VAL: формулировка x режим рассуждения (test не трогаем)
    tuned = []
    for variant in args.variants.split(","):
        for mode in modes:
            tuned.append(run_one(model, val, pool, args, tune_k, variant, mode, "val", "tune"))
    top = max(tuned, key=lambda r: (round(r["macro_f1"], 4), -r["lat_p50"]))
    print(f"Выбрана конфигурация по VAL: variant={top['variant']}, reasoning={top['reasoning']}")
    # 2) финальный прогон на TEST: базовая и улучшенная конфигурации + все формулировки (чувствительность)
    final = run_one(model, test, pool, args, tune_k, top["variant"], top["reasoning"], "test", "final")
    for variant in args.variants.split(","):
        run_one(model, test, pool, args, tune_k, variant, "direct", "test", "sens")
    (Path(args.out) / "best.json").write_text(json.dumps(
        {"model": model, "shots": tune_k, "variant": top["variant"], "reasoning": top["reasoning"],
         "test_macro_f1": final["macro_f1"], "ci": [final["macro_f1_lo"], final["macro_f1_hi"]]}, ensure_ascii=False, indent=1))

def md(rows, cols, head):
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    return s + "".join("| " + " | ".join(str(c(r)) for c in cols) + " |\n" for r in rows)

def stage_report(args):
    runs = read_runs(args); out = Path(args.out); text = ["# Результаты задания 3\n"]
    test = [r for r in runs if r["split"] == "test" and r["reasoning"] == "direct"]
    sel = [r for r in test if r["variant"] == args.variant]
    sel.sort(key=lambda r: (-r["macro_f1"], r["lat_p50"]))
    text += ["## 1. Сравнение моделей (test, одна формулировка, temperature=%s, seed=%s)\n" % (args.temperature, args.seed),
             md(sel, [lambda r: r["model"], lambda r: "zero-shot" if r["shots"] == 0 else f"{r['shots']}-shot",
                      lambda r: f"{r['macro_f1']:.3f} [{r['macro_f1_lo']:.3f}; {r['macro_f1_hi']:.3f}]",
                      lambda r: f"{r['accuracy']:.3f}", lambda r: f"{r['mcc']:.3f}",
                      lambda r: f"{100 * r['invalid_rate']:.1f}%", lambda r: f"{r['lat_p50']:.2f}/{r['lat_p95']:.2f}",
                      lambda r: r["prompt_tokens"] + r["completion_tokens"]],
                ["Модель", "Режим", "Macro-F1 [95% CI]", "Accuracy", "MCC", "Невалидные", "Латентность p50/p95, с", "Токены"])]
    sens = defaultdict(list)
    for r in test:
        sens[(r["model"], r["shots"])].append(r)
    rows = []
    for (m, k), rs in sens.items():
        if len(rs) > 1:
            f = [r["macro_f1"] for r in rs]
            rows.append((m, k, len(rs), min(f), max(f), statistics.pstdev(f)))
    if rows:
        text += ["\n## 2. Чувствительность к формулировке инструкции (test)\n",
                 md(rows, [lambda r: r[0], lambda r: r[1], lambda r: r[2], lambda r: f"{r[3]:.3f}",
                           lambda r: f"{r[4]:.3f}", lambda r: f"{r[4] - r[3]:.3f}", lambda r: f"{r[5]:.3f}"],
                    ["Модель", "k", "Формулировок", "Min F1", "Max F1", "Размах", "Std"])]
    tune = sorted([r for r in runs if r["split"] == "val"], key=lambda r: -r["macro_f1"])
    if tune:
        text += ["\n## 3. Тюнинг лучшей модели (VAL)\n",
                 md(tune, [lambda r: r["model"], lambda r: r["variant"], lambda r: r["reasoning"],
                           lambda r: f"{r['macro_f1']:.3f} [{r['macro_f1_lo']:.3f}; {r['macro_f1_hi']:.3f}]",
                           lambda r: f"{100 * r['invalid_rate']:.1f}%", lambda r: f"{r['lat_p50']:.2f}"],
                    ["Модель", "Формулировка", "Рассуждение", "Macro-F1 [95% CI]", "Невалидные", "p50, с"])]
    if (out / "best.json").exists():
        b = json.loads((out / "best.json").read_text())
        text.append(f"\n**Итоговая конфигурация:** {b['model']}, k={b['shots']}, формулировка {b['variant']}, "
                    f"рассуждение {b['reasoning']}; test macro-F1 = {b['test_macro_f1']:.3f} "
                    f"[{b['ci'][0]:.3f}; {b['ci'][1]:.3f}]\n")
        pc = next((r for r in runs if r["split"] == "test" and r["run_id"].startswith("test__" + b["model"].replace("/", "_"))
                   and r["variant"] == b["variant"] and r["reasoning"] == b["reasoning"] and r["shots"] == b["shots"]), None)
        if pc:
            text += ["\n## 4. F1 по классам (итоговая конфигурация, test)\n",
                     md(sorted(pc["per_class_f1"].items(), key=lambda x: x[1]), [lambda r: r[0], lambda r: r[1]], ["Класс", "F1"])]
    text += ["\n## 5. Журнал параметров\n",
             md(sorted(runs, key=lambda r: r["ts"]),
                [lambda r: r["ts"][5:16], lambda r: r["run_id"], lambda r: r["temperature"], lambda r: r["seed"],
                 lambda r: r["max_tokens"], lambda r: r["n"], lambda r: r["test_sha"], lambda r: r["script_sha"],
                 lambda r: ",".join(r["dropped_params"]) or "—"],
                ["Время", "Прогон", "T", "seed", "max_tok", "N", "sha test", "sha script", "не поддерж. параметры"])]
    s = "".join(text)
    (out / "report.md").write_text(s, encoding="utf-8")
    keys = [k for k in runs[0] if k != "per_class_f1"] if runs else []
    with open(out / "metrics_all.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(runs)
    print(s); print(f"\nСохранено: {out/'report.md'}, {out/'metrics_all.csv'}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["zero", "few", "best", "report"])
    ap.add_argument("--list-models", action="store_true", help="показать id моделей, доступных у прокси")
    ap.add_argument("--data", default="out", help="папка результата prepare_dataset.py")
    ap.add_argument("--out", default="llm_out")
    ap.add_argument("--models", type=lambda s: s.split(","), default=DEFAULT_MODELS)
    ap.add_argument("--model", help="stage best: принудительно выбрать модель")
    ap.add_argument("--shots", type=int, default=3, help="число few-shot примеров (2-3)")
    ap.add_argument("--variant", default="v0", choices=["v0", "v1", "v2"], help="формулировка для основного сравнения")
    ap.add_argument("--variants", default="v0,v1,v2", help="stage best: формулировки для тюнинга/чувствительности")
    ap.add_argument("--reasoning-modes", default="direct,cot,effort_low,effort_medium,effort_high",
                    help="direct | cot (краткое рассуждение в JSON) | effort_* (параметр reasoning_effort)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--boot", type=int, default=1000, help="число bootstrap-повторов для 95%% CI")
    ap.add_argument("--limit", type=int, default=0, help="для пробы: взять N наблюдений (случайно, с seed)")
    ap.add_argument("--hard", action="store_true", help="few-shot: исключать примеры с тем же заголовком, что у теста")
    ap.add_argument("--mock", action="store_true", help="без сети: имитация ответов, проверка пайплайна")
    args = ap.parse_args()
    random.seed(args.seed)
    d = Path(args.data)
    args.train, args.val, args.test = (d / f"observations_{s}.csv" for s in ("train", "val", "test"))
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.list_models:
        r = requests.get(f"{API_BASE}/models", headers={"Authorization": f"Bearer {os.environ.get('PROXY_API_KEY', '')}"}, timeout=30)
        print("\n".join(sorted(m["id"] for m in r.json().get("data", [])))); return
    if not args.stage:
        ap.error("укажите --stage")
    if args.stage == "zero":
        stage_grid(args, "zero", 0)
    elif args.stage == "few":
        stage_grid(args, "few", args.shots)
    elif args.stage == "best":
        stage_best(args)
    else:
        stage_report(args)

if __name__ == "__main__":
    main()