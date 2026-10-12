"""Усложнение тестового набора: синтетические искажения ПОСЛЕ разбиения train/val/test.

Берёт out/observations_{train,val,test}.csv (результат prepare_dataset.py) и пишет
out_hard/observations_{train,val,test}.csv в том же формате, поэтому
    python run_llm_benchmark.py --data out_hard --out llm_out_hard --stage zero
работает без изменений.

Что делается внутри КАЖДОЙ выборки отдельно (утечки между выборками нет):
  1. реальные наблюдения дедуплицируются по (label, header_text) и остаются как есть (recipe=real);
  2. для каждого строится --per-obs искажённых вариантов (опечатки, перенос слов, регистр,
     сокращения, английский/транслит, префикс «Графа N:», форматы чисел, пустой/«Колонка N» заголовок);
  3. добавляются полностью синтетические колонки из словарей синонимов (--synth-per-class на класс)
     и «отвлекающие» колонки класса unmapped (Примечание, Вес, Серийный номер...).
Словари синонимов делятся seed'ом на общие (shared) и свои для каждой выборки: test содержит
формулировки, которых нет в train. Рецепты из --holdout-recipes в train не попадают (для QLoRA).
В конце выводится отчёт о пересечениях train/test по (header_text, values_sample).
"""
import argparse, csv, json, random, re
from collections import Counter
from pathlib import Path

SYN = {
    "item_code": ["Код товара", "Артикул", "Код ТРУ", "Код по КТРУ", "Код позиции", "ОКПД2", "Код номенклатуры", "SKU", "Арт.", "Идентификатор товара"],
    "row_number": ["№ п/п", "№", "Номер по порядку", "N п/п", "Поз.", "Номер строки", "№ пп", "No.", "Порядковый номер", "Строка"],
    "estimate_position": ["Номер позиции по смете", "Позиция сметы", "Поз. по смете", "№ в смете", "Шифр сметы", "Пункт сметы", "Код сметной позиции", "Смета, поз."],
    "item_name": ["Наименование товара (описание выполненных работ, оказанных услуг)", "Наименование", "Товар", "Название товара", "Наименование работ/услуг", "Предмет закупки", "Описание", "Item", "Номенклатура", "Продукция"],
    "unit": ["Единица измерения", "Ед. изм.", "Ед.изм", "Измерение", "Unit", "Единицы", "ЕИ", "Ед."],
    "unit_code": ["Код единицы измерения", "Код ОКЕИ", "Код ед. изм.", "ОКЕИ код", "Цифровой код ЕИ", "Код по ОКЕИ"],
    "unit_name": ["Условное обозначение (национальное)", "Обозначение ед. изм.", "Усл. обозначение", "Краткое наименование ЕИ", "Наименование единицы измерения"],
    "quantity": ["Количество (объем)", "Кол-во", "Количество", "Объём", "Qty", "Колич.", "Число единиц", "Объем поставки", "Количество, шт."],
    "price": ["Цена (тариф) за единицу измерения", "Цена", "Цена за ед.", "Тариф", "Price", "Стоимость единицы", "Цена единицы, руб.", "Единичная цена"],
    "amount_without_vat": ["Стоимость товаров (работ, услуг) без налога - всего", "Сумма без НДС", "Стоимость без налога", "Итого без НДС", "Сумма (нетто)", "Цена без НДС всего", "Amount excl. VAT", "Всего без налога"],
    "excise": ["В том числе сумма акциза", "Акциз", "Сумма акциза", "Excise", "Акцизный сбор", "Акциз, руб."],
    "vat_rate": ["Налоговая ставка", "Ставка НДС", "НДС, %", "Ставка налога", "VAT rate", "% НДС", "Ставка, %", "Налог, ставка"],
    "vat_amount": ["Сумма налога, предъявляемая покупателю", "Сумма НДС", "НДС, руб.", "Налог", "VAT amount", "Сумма налога", "Размер НДС", "НДС сумма"],
    "total_with_vat": ["Стоимость товаров (работ, услуг) с налогом - всего", "Сумма с НДС", "Итого с НДС", "Всего к оплате", "Стоимость с налогом", "Сумма (брутто)", "Total incl. VAT", "К оплате"],
    "country_of_origin": ["Страна происхождения товара", "Страна происх.", "Страна производитель", "Происхождение", "Country of origin", "Страна", "Страна изготовления", "Код страны происхождения"],
    "unmapped": ["Примечание", "Серийный номер", "Вес, кг", "Дата поставки", "Срок годности", "Комментарий", "Номер партии", "Склад", "Производитель", "Гарантия, мес.", "Артикул поставщика", "Штрихкод"],
}
GOODS = ["Молоко 3,2%", "Крупа гречневая", "Услуги охраны помещений", "Мыло хозяйственное", "Бумага А4", "Ремонт кровли", "Яйцо куриное", "Хлеб пшеничный", "Моющее средство", "Вывоз ТКО"]
UNITS = [("шт", "796"), ("кг", "166"), ("усл. ед", "876"), ("л", "112"), ("уп", "778"), ("мес", "362")]

def vals(label, r):
    p = round(r.uniform(5, 40000), 2); q = r.choice([1, 2, 5, 10, 12.5, 100]); s = round(p * q, 2)
    rate = r.choice(["20%", "10%", "5%", "без НДС", "0%"]); vat = round(s * 0.05, 2)
    u = r.choice(UNITS)
    f = {"item_code": lambda: " | ".join(f"{r.randint(10,99)}.{r.randint(10,99)}.{r.randint(10,99)}" for _ in range(3)),
         "row_number": lambda: "1 | 2 | 3", "estimate_position": lambda: " | ".join(str(r.randint(1, 40)) for _ in range(3)),
         "item_name": lambda: " | ".join(r.sample(GOODS, 3)), "unit": lambda: f"{u[1]} {u[0]}", "unit_code": lambda: u[1],
         "unit_name": lambda: u[0], "quantity": lambda: f"{q} | {r.choice([1,3,8])} | {r.choice([20,50])}",
         "price": lambda: f"{p} | {round(p/3,2)}", "amount_without_vat": lambda: f"{s} | {round(s/2,2)}",
         "excise": lambda: r.choice(["без акциза", "X", "0.00"]), "vat_rate": lambda: f"{rate} | {rate}",
         "vat_amount": lambda: f"{vat} | {round(vat/2,2)}", "total_with_vat": lambda: f"{round(s+vat,2)} | {round((s+vat)/2,2)}",
         "country_of_origin": lambda: r.choice(["РОССИЯ", "643 РОССИЯ", "Россия | Россия", "КИТАЙ"]),
         "unmapped": lambda: r.choice(["", "-", "б/н | 12/2026", "ок", "2026-05-14", "0.5"])}
    return f[label]()

def fmt_num(v, r):
    def one(x):
        if re.fullmatch(r"-?\d+(\.\d+)?", x):
            c = r.choice(["comma", "space", "rub", "same"])
            return {"comma": x.replace(".", ","), "space": f"{float(x):,.2f}".replace(",", " "), "rub": x + " руб.", "same": x}[c]
        return x
    return " | ".join(one(x.strip()) for x in v.split(" | "))

ABBR = [("Количество", "Кол-во"), ("Наименование", "Наим."), ("Единица измерения", "Ед. изм."), ("Стоимость", "Ст-ть"), ("налог", "нал."), ("товаров", "тов."), ("Страна", "Стр.")]
TRANSLIT = [("Цена", "Cena"), ("Количество", "Kolichestvo"), ("Наименование", "Naimenovanie"), ("Сумма", "Summa"), ("Страна", "Strana"), ("налог", "nalog"), ("Единица", "Edinica")]
EN = [("Цена", "Price"), ("Количество", "Quantity"), ("Наименование", "Name"), ("Сумма", "Amount"), ("Страна", "Country"), ("налог", "tax"), ("Единица измерения", "Unit")]

def typo(s, r):
    if len(s) < 4: return s
    i = r.randrange(1, len(s) - 1); k = r.choice(["del", "swap", "dup"])
    return s[:i] + s[i + 1:] if k == "del" else s[:i] + s[i + 1] + s[i] + s[i + 2:] if k == "swap" else s[:i] + s[i] + s[i:]

def subst(s, table):
    for a, b in table:
        if a.lower() in s.lower():
            return re.sub(re.escape(a), b, s, flags=re.I)
    return s

def distort(h, v, recipe, r):
    if recipe == "typo": return typo(h, r), v
    if recipe == "case": return r.choice([h.upper(), h.lower()]), v
    if recipe == "hyphen":
        w = max(h.split(), key=len); i = len(w) // 2
        return h.replace(w, w[:i] + "-" + w[i:], 1), v
    if recipe == "abbr": return subst(h, ABBR), v
    if recipe == "punct": return re.sub(r"[()\-,./]", " ", h), v
    if recipe == "prefix": return f"Графа {r.randint(1, 15)}: {h}", v
    if recipe == "english": return subst(h, EN), v
    if recipe == "translit": return subst(h, TRANSLIT), v
    if recipe == "numfmt": return h, fmt_num(v, r)
    if recipe == "noheader": return r.choice(["Колонка %d" % r.randint(1, 15), "Графа %d" % r.randint(1, 15), "Column", "—"]), v
    if recipe == "combo": return typo(subst(h, ABBR), r).upper(), fmt_num(v, r)
    raise ValueError(recipe)

RECIPES = ["typo", "case", "hyphen", "abbr", "punct", "prefix", "english", "translit", "numfmt", "noheader", "combo"]

def read(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def family(seed):
    """Делит синонимы каждого класса: ~50% общие, остальное поровну между train/val/test."""
    r = random.Random(seed); out = {s: {} for s in ("train", "val", "test")}
    for lab, syns in SYN.items():
        syns = syns[:]; r.shuffle(syns); cut = max(2, len(syns) // 2)
        shared, rest = syns[:cut], syns[cut:]
        for i, s in enumerate(("train", "val", "test")):
            out[s][lab] = shared + rest[i::3]
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="out"); ap.add_argument("--out", default="out_hard")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-obs", type=int, default=6, help="искажённых вариантов на каждое реальное наблюдение")
    ap.add_argument("--synth-per-class", type=int, default=12, help="синтетических колонок на класс в каждой выборке")
    ap.add_argument("--holdout-recipes", default="english,translit,prefix", help="рецепты, которых нет в train")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    fam = family(a.seed); hold = set(filter(None, a.holdout_recipes.split(",")))
    fields = ["obs_id", "split", "cluster_id", "header_text", "values_sample", "label", "recipe", "source_obs"]
    res = {}
    for si, sp in enumerate(("train", "val", "test")):
        r = random.Random(f"{a.seed}:{sp}")
        recipes = [x for x in RECIPES if not (sp == "train" and x in hold)]
        base, seen = [], set()
        for o in read(Path(a.data) / f"observations_{sp}.csv"):
            k = (o["label"], o["header_text"].lower())
            if k not in seen:
                seen.add(k); base.append(o)
        rows = []; n = 0
        def add(cl, h, v, lab, rec, src):
            nonlocal n; n += 1
            rows.append(dict(obs_id=f"{sp}_h{n}", split=sp, cluster_id=cl, header_text=h, values_sample=v, label=lab, recipe=rec, source_obs=src))
        for o in base:
            add(o["cluster_id"], o["header_text"], o["values_sample"], o["label"], "real", o["obs_id"])
            for rec in r.sample(recipes, min(a.per_obs, len(recipes))):
                h, v = distort(o["header_text"], o["values_sample"], rec, r)
                add(o["cluster_id"], h, v, o["label"], rec, o["obs_id"])
        for lab in SYN:
            for j in range(a.synth_per_class):
                h = r.choice(fam[sp][lab]); v = vals(lab, r); rec = "synth"
                if r.random() < 0.6:
                    rec = r.choice(recipes); h, v = distort(h, v, rec, r); rec = "synth+" + rec
                add(f"synth_{sp}_{lab}_{j // 4}", h, v, lab, rec, "")
        res[sp] = rows
    # удаляем из val/test точные совпадения с train, чтобы few-shot не получал готовый ответ
    tr = {(x["header_text"].lower(), x["values_sample"]) for x in res["train"]}
    trh = {x["header_text"].lower() for x in res["train"]}
    rep = {}
    for sp in ("val", "test"):
        before = len(res[sp]); res[sp] = [x for x in res[sp] if (x["header_text"].lower(), x["values_sample"]) not in tr]
        rep[sp] = {"removed_exact_with_train": before - len(res[sp]), "n": len(res[sp]),
                   "header_overlap_with_train": round(sum(x["header_text"].lower() in trh for x in res[sp]) / len(res[sp]), 3)}
    for sp, rows in res.items():
        with open(out / f"observations_{sp}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
        rep.setdefault(sp, {})["n"] = len(rows)
        rep[sp]["by_label"] = dict(Counter(x["label"] for x in rows))
        rep[sp]["by_recipe"] = dict(Counter(x["recipe"].split("+")[-1] for x in rows))
    rep["holdout_recipes_absent_in_train"] = sorted(hold)
    (out / "hard_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for sp in ("train", "val", "test"):
        print(sp, rep[sp]["n"], {k: v for k, v in rep[sp].items() if k not in ("by_label", "by_recipe", "n")})
    print("Сохранено в", out)

if __name__ == "__main__":
    main()
