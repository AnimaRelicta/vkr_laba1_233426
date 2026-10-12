"""Достаёт .html документы из zip-архивов и сохраняет их под именем архива.

Пример:
    data/UPD_20260213_2780702609525000029.zip
        -> UPD_20260206_019C..._BCE4....html (внутри архива)
    upd/UPD_20260213_2780702609525000029.html (результат)

Запуск:
    python extract_upd.py                      # data -> upd, только архивы UPD_*
    python extract_upd.py -i data -o upd
    python extract_upd.py --prefix ACT_        # другой тип документов
    python extract_upd.py --prefix ""          # все архивы
"""
import argparse
import zipfile
from pathlib import Path


def extract_html(zip_path: Path, out_dir: Path, overwrite: bool) -> int:
    """Сохраняет html из одного архива. Возвращает число сохранённых файлов."""
    saved = 0
    with zipfile.ZipFile(zip_path) as zf:
        htmls = [i for i in zf.infolist()
                 if not i.is_dir() and i.filename.lower().endswith((".html", ".htm"))]
        if not htmls:
            print(f"[нет html] {zip_path.name}")
            return 0
        for n, info in enumerate(htmls, start=1):
            # один html -> имя архива; несколько -> добавляем _2, _3 ...
            suffix = "" if n == 1 else f"_{n}"
            target = out_dir / f"{zip_path.stem}{suffix}.html"
            if target.exists() and not overwrite:
                print(f"[пропуск, уже есть] {target.name}")
                continue
            target.write_bytes(zf.read(info))
            print(f"[ok] {zip_path.name} -> {target.name}")
            saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Извлечение html из zip-архивов")
    parser.add_argument("-i", "--input", default="data", help="папка с архивами (по умолчанию data)")
    parser.add_argument("-o", "--output", default="upd", help="папка для результата (по умолчанию upd)")
    parser.add_argument("--prefix", default="UPD_", help='брать только архивы с этим началом имени; "" = все')
    parser.add_argument("--overwrite", action="store_true", help="перезаписывать существующие файлы")
    args = parser.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    if not in_dir.is_dir():
        raise SystemExit(f"Папка не найдена: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(p for p in in_dir.rglob("*.zip") if p.name.startswith(args.prefix))
    total = bad = 0
    for zip_path in archives:
        try:
            total += extract_html(zip_path, out_dir, args.overwrite)
        except zipfile.BadZipFile:
            bad += 1
            print(f"[битый архив] {zip_path.name}")

    print(f"\nАрхивов: {len(archives)}, сохранено html: {total}, битых: {bad}")


if __name__ == "__main__":
    main()