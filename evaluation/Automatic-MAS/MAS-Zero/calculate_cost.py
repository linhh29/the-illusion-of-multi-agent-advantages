#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Tuple


def find_files(patterns: List[str]) -> List[Path]:
    files = []
    for pat in patterns:
        # recursive=True 以支持 ** 通配
        files.extend(Path(p).resolve() for p in glob.glob(pat, recursive=True))
    # 去重并仅保留文件
    uniq = [p for p in sorted(set(files)) if p.is_file()]
    return uniq


def load_json_list(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"读取 JSON 失败: {path} -> {e}") from e
    if not isinstance(data, list):
        raise ValueError(f"文件不是 List 格式: {path}")
    return data


def to_decimal(value) -> Decimal:
    # 通过 str 转换，避免 float 精度误差
    if isinstance(value, (int, float, str)):
        try:
            return Decimal(str(value))
        except InvalidOperation as e:
            raise ValueError(f"无法解析为 Decimal 的数值: {value!r}") from e
    raise TypeError(f"不支持的 total_cost 类型: {type(value)}")


def sum_cost_in_file(path: Path, ignore_missing: bool) -> Tuple[Decimal, int, int]:
    """
    返回: (该文件总cost, 该文件item总数, 被跳过的item数)
    """
    data = load_json_list(path)
    total = Decimal("0")
    skipped = 0
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            if ignore_missing:
                skipped += 1
                print(f"[WARN] 非字典项已跳过: {path}#{idx}", file=sys.stderr)
                continue
            raise ValueError(f"列表项不是字典: {path}#{idx}")
        if "total_cost" not in item:
            if ignore_missing:
                skipped += 1
                print(f"[WARN] 缺少 total_cost，已跳过: {path}#{idx}", file=sys.stderr)
                continue
            raise KeyError(f"缺少 total_cost 字段: {path}#{idx}")
        try:
            total += to_decimal(item["total_cost"])
        except Exception as e:
            if ignore_missing:
                skipped += 1
                print(f"[WARN] 无法解析 total_cost，已跳过: {path}#{idx} -> {e}", file=sys.stderr)
                continue
            raise
    return total, len(data), skipped


def main():
    parser = argparse.ArgumentParser(
        description="累计多个 JSON 文件（每个文件为 List），把每个 item 的 total_cost 相加。"
    )
    parser.add_argument("patterns", nargs="+", help="一个或多个 glob 模式，例如 'logs/**/*.json'")
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="忽略缺失/非法的 total_cost（默认遇到即报错）",
    )
    args = parser.parse_args()

    files = find_files(args.patterns)
    if not files:
        print("未匹配到任何文件。", file=sys.stderr)
        sys.exit(2)

    grand_total = Decimal("0")
    total_items = 0
    total_skipped = 0

    for path in files:
        file_total, n_items, skipped = sum_cost_in_file(path, args.ignore_missing)
        grand_total += file_total
        total_items += n_items
        total_skipped += skipped

    # 输出结果
    print(f"匹配到文件数: {len(files)}")
    print(f"总条目数: {total_items}")
    if args.ignore_missing and total_skipped:
        print(f"被跳过条目数: {total_skipped}")
    # 直接输出数值；如需单位可自行在外部加上货币符号
    print(f"总计 total_cost: {grand_total}")


if __name__ == "__main__":
    main()
