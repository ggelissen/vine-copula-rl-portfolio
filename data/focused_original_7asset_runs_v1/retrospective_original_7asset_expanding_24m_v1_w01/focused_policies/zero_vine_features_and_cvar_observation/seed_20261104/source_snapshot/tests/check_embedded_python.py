#!/usr/bin/env python3
"""Compile literal Python blocks embedded in R py_run_string() calls."""
from pathlib import Path


def decode_r_string(value: str) -> str:
    mapping = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}
    output, index = [], 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value) and value[index + 1] in mapping:
            output.append(mapping[value[index + 1]])
            index += 2
        else:
            output.append(value[index])
            index += 1
    return "".join(output)


def blocks(path: Path):
    text, position, marker = path.read_text(encoding="utf-8"), 0, "py_run_string("
    while True:
        start = text.find(marker, position)
        if start < 0:
            return
        quote = start + len(marker)
        while text[quote].isspace():
            quote += 1
        if text[quote] != '"':
            raise ValueError(f"{path}: py_run_string argument is not a literal")
        end, escaped = quote + 1, False
        while end < len(text):
            char = text[end]
            if char == '"' and not escaped:
                break
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
            end += 1
        yield decode_r_string(text[quote + 1 : end])
        position = end + 1


count = 0
for r_file in (
    Path("rl/train_rl.r"),
    Path("rl/evaluate_rl.r"),
    Path("rl/training_sanity_check.r"),
):
    for number, source in enumerate(blocks(r_file), 1):
        compile(source, f"{r_file}:py_run_string#{number}", "exec")
        count += 1
print(f"Compiled {count} embedded Python blocks.")
