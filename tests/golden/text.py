#!/usr/bin/env python3
"""text.py: golden data for athanor's text tests (tests/test_text.c). Development only.

Three kinds of records, all produced by the pinned reference (transformers
5.24.0 with its tokenizers backend, the checkpoint's own tokenizer.json and
chat_template.jinja):

  - corpus: texts and the token ids the reference tokenizer gives them, and
    the text it decodes the ids back to. Hand-written cases (languages,
    emoji, code, white space, contractions, added tokens in text,
    normalization), files of this repository, and two sweeps: every Unicode
    code point in a few contexts, and every canonical decomposition with
    its marks shuffled. athanor must give the same ids and the same text;
  - templates: conversations, with and without tools, thinking on and off,
    rendered by the reference's apply_chat_template, byte for byte, and
    tokenized. Conversations the template rejects are recorded as errors;
  - generate: a chat prompt, greedy decoding by the fp32 reference (the
    exact model) with the top-2 margin of every step, and the bf16 noise of
    that margin (its largest change from fp32 to bf16 on the same tokens).

    tests/golden/.venv/bin/python tests/golden/text.py models/Qwen3.5-0.8B

writes tests/golden/data/qwen35-0.8b/{text.safetensors,text.json}.
"""

import argparse
import importlib.util
import json
import os
import platform
import random
import sys
import unicodedata

import numpy as np
import torch
import transformers
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

spec = importlib.util.spec_from_file_location("dump", os.path.join(HERE, "dump.py"))
dump = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dump)

CASES = [
    ("empty", ""),
    ("ascii", "Hello, world! This is a test of the tokenizer."),
    ("italian", "Perché l'università è chiusa? Città, caffè e però: àèìòù ÀÈÌÒÙ."),
    ("german", "Größere Straßen über Brücken; Ärger, Öl, Übel. ẞ"),
    ("french", "L'élève a dû aller à l'hôpital — c'était grave ? « Oui » !"),
    ("spanish", "¿Dónde está el niño? ¡Allí! Mañana será otro día."),
    ("russian", "Съешь же ещё этих мягких французских булок, да выпей чаю."),
    ("greek", "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία. ΆΈΉΊΌΎΏ ΐΰ"),
    ("arabic", "نص حكيم له سر قاطع وذو شأن عظيم مكتوب على ثوب أخضر ومغلف بجلد أزرق"),
    ("hebrew", "דג סקרן שט בים מאוכזב ולפתע מצא חברה"),
    ("hindi", "ऋषियों को सताने वाले दुष्ट राक्षसों के राजा रावण का सर्वनाश करने वाले विष्णुवतार भगवान श्रीराम"),
    ("chinese", "我能吞下玻璃而不伤身体。你好，世界！人工智能（AI）正在改变世界。"),
    ("japanese", "いろはにほへと ちりぬるを わかよたれそ つねならむ。カタカナ、ｶﾀｶﾅ、漢字。"),
    ("korean", "다람쥐 헌 쳇바퀴에 타고파. 한국어 텍스트입니다."),
    ("thai", "เป็นมนุษย์สุดประเสริฐเลิศคุณค่า กว่าบรรดาฝูงสัตว์เดรัจฉาน"),
    ("vietnamese", "Tiếng Việt có dấu: Trăm năm trong cõi người ta, chữ tài chữ mệnh khéo là ghét nhau."),
    ("emoji", "😀 👍🏽 👨‍👩‍👧‍👦 🇮🇹🇯🇵 ❤️ 🏳️‍🌈 #️⃣ 🧑🏿‍💻 ok👌"),
    ("numbers", "3.14159 2,718 1e-10 0x1F -42 +7 1234567890 ١٢٣ Ⅻⅻ ½ ²³ 𝟙𝟚"),
    ("contractions", "I'm you're he's we've they'll I'd can't IT'S THEY'RE We'VE 'tis 'Twas 'ſure x'ſ 'Kelvin"),
    ("apostrophes", "don’t ’s ' s '' 's's 'S'T'RE'VE'M'LL'D"),
    ("whitespace", "a  b   c\t\td\n\ne\r\nf \n g h　i j k\u0085l᠎m​n \t \n\n  \n"),
    ("trailing_space", "end with spaces   "),
    ("leading_space", "   start with spaces"),
    ("only_spaces", "     "),
    ("only_newlines", "\n\n\n"),
    ("crlf", "line one\r\nline two\r\n\r\nline four\r"),
    ("tabs_code", "\tif (x) {\n\t\treturn y;\n\t}\n"),
    ("python", "def f(x: int) -> int:\n    '''Doc.'''\n    return x ** 2  # square\n\n\nprint(f(3))\n"),
    ("c_code", "int main(void) {\n    printf(\"%d\\n\", 42);\n    return 0;\n}\n"),
    ("json", '{"key": [1, 2.5, true, null], "nested": {"a": "b\\n"}}'),
    ("html", "<div class=\"x\">&amp; <b>bold</b></div><!-- c -->"),
    ("url", "https://example.com/path?query=1&b=2#frag mailto:a@b.c"),
    ("punctuation", "!!!??? ... --- ___ *** ### @@@ $$$ %%% ^^^ &&& ((( ))) [[[ ]]] {{{ }}} ||| \\\\\\ /// ~~~ ```"),
    ("mixed_scripts", "abcабвαβγאבגابت中文かなカナ한글"),
    ("marks", "é à́ ́start x̣̂ q̣̇ 각 Å Ω क़"),
    ("nfc_needed", "Café naïve Amélie Zürich"),
    ("special_tokens", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nhello<|endoftext|>"),
    ("special_partial", "<|im_start <|im_end| <think <tool_call>x</tool_call> <|vision_start|><|image_pad|>"),
    ("tool_markup", "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"),
    ("long_word", "a" * 300 + " " + "ab" * 200 + " " + "0" * 50),
    ("long_spaces", " " * 100 + "x" + "\n" * 40 + "y" + "\t" * 30),
    ("control", "a\x00b\x01c\x7fd\x1be[0m"),
    ("private_use", "\U000f0000\U0010fffd"),
    ("unassigned", "͸͹ \U0001fbff \U000e01f0"),
]

REPO_FILES = ["README.md", "AGENTS.md", "docs/MODELS.md", "src/core/json.c", "tests/golden/dump.py"]


def all_code_points_cases():
    """Every code point but the surrogates, each in a few contexts, in chunks."""
    cps = [c for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF)]
    out = []
    for i in range(0, len(cps), 8192):
        parts = []
        for c in cps[i:i + 8192]:
            ch = chr(c)
            parts.append(f"a{ch} {ch}1{ch}!{ch}'{ch}\n")
        out.append((f"codepoints_{cps[i]:06x}", "".join(parts)))
    return out


def decomposition_cases():
    """Every code point with a canonical decomposition, decomposed, with
    extra marks in shuffled order: NFC must recompose and reorder."""
    rng = random.Random(1234)
    marks = ["́", "̀", "̣", "̈", "̧", "̛", "ͅ", "ְ", "़", "゙"]
    parts = []
    for c in range(0x110000):
        if 0xD800 <= c <= 0xDFFF:
            continue
        d = unicodedata.decomposition(chr(c))
        if not d or d.startswith("<"):
            continue
        extra = rng.sample(marks, rng.randint(0, 3))
        parts.append(unicodedata.normalize("NFD", chr(c)) + "".join(extra) + " ")
    hangul = "".join(unicodedata.normalize("NFD", chr(c)) for c in range(0xAC00, 0xD7A4, 7))
    return [("decompositions", "".join(parts)), ("hangul_jamo", hangul)]


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. \"Paris\""},
                "days": {"type": "integer", "minimum": 1, "maximum": 14},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Cerca nei file: «pattern» con\tregex.\nRestituisce righe — e altro \u0001.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "limit": {"type": ["integer", "null"], "default": 10},
                "score": {"type": "number", "minimum": 0.5, "maximum": 1e-7, "multipleOf": 0.001},
                "recursive": {"type": "boolean"},
                "paths": {"type": "array", "items": {"type": "string"}},
                "options": {"type": "object", "properties": {"case": {"type": "boolean"}}},
            },
            "required": ["pattern"],
        },
    },
}
NO_PARAMS = {"type": "function", "function": {"name": "now", "description": "The time.", "parameters": {"type": "object", "properties": {}}}}


def call(name, args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


TEMPLATES = [
    ("user", [{"role": "user", "content": "Hello!"}], None, True, False),
    ("user_thinking", [{"role": "user", "content": "Hello!"}], None, True, True),
    ("user_no_generate", [{"role": "user", "content": "Hello!"}], None, False, False),
    ("system_user", [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Why is the sky blue?"}], None, True, False),
    ("trim", [{"role": "system", "content": "  \n sys 　"}, {"role": "user", "content": "\t hi there \n\n"}], None, True, False),
    ("empty_system", [{"role": "system", "content": ""}, {"role": "user", "content": "x"}], None, True, False),
    ("multi_turn", [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "And 3+3?"},
    ], None, True, True),
    ("history_think_in_content", [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "<think>\nreasoning one\n</think>\n\nAnswer one."},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "<think>\n\nreasoning two\n\n</think>\n\n\nAnswer two."},
    ], None, False, False),
    ("history_reasoning_field", [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1 </think> kept", "reasoning_content": "  thought  "},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2", "reasoning_content": "\nsecond thought\n"},
    ], None, True, False),
    ("tools", [{"role": "user", "content": "Weather in Paris?"}], [WEATHER, SEARCH, NO_PARAMS], True, False),
    ("tools_system", [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Rome?"}], [WEATHER], True, True),
    ("tools_empty_system", [{"role": "system", "content": "   "}, {"role": "user", "content": "x"}], [NO_PARAMS], True, False),
    ("tool_round_trip", [
        {"role": "system", "content": "You can call tools."},
        {"role": "user", "content": "Weather in Paris and Rome for 3 days?"},
        {"role": "assistant", "content": "Let me check.", "reasoning_content": "Two cities, call twice.",
         "tool_calls": [call("get_weather", {"city": "Paris", "days": 3}), call("get_weather", {"city": "Rome", "days": 3, "unit": "celsius"})]},
        {"role": "tool", "content": "{\"temp\": 18}", "tool_call_id": "c1"},
        {"role": "tool", "content": "  sunny, 24°C  ", "tool_call_id": "c2"},
        {"role": "assistant", "content": "Paris 18°C, Rome 24°C."},
        {"role": "user", "content": "Thanks!"},
    ], [WEATHER], True, False),
    ("tool_call_no_content", [
        {"role": "user", "content": "Search."},
        {"role": "assistant", "content": "", "tool_calls": [call("search_files", {
            "pattern": "a\"b\\c\n", "limit": 5, "score": 0.25, "recursive": True, "paths": ["src", "ü"],
            "options": {"case": False, "x": None}, "big": 1e20, "small": 1e-05, "neg": -0.0, "n": None, "f": 2.0})]},
        {"role": "tool", "content": "found"},
    ], [SEARCH], True, True),
    ("tool_response_as_user", [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "calling", "tool_calls": [call("now", {})]},
        {"role": "user", "content": "<tool_response>\n12:00\n</tool_response>"},
    ], [NO_PARAMS], True, False),
    ("tool_first", [{"role": "tool", "content": "orphan"}, {"role": "user", "content": "hi"}], None, True, False),
    ("unicode", [{"role": "user", "content": "Ciao 👋 — «come stai?» 你好"}], None, True, False),
    ("error_no_user", [{"role": "system", "content": "only system"}], None, True, False),
    ("error_system_late", [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}], None, True, False),
    ("error_only_tool_responses", [{"role": "user", "content": "<tool_response>x</tool_response>"}], None, True, False),
]

GENERATE_PROMPT = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
GENERATE_STEPS = 32


def c_view(messages, tools):
    """The conversation as athanor's ath_msg and ath_tool_def take it:
    arguments and schemas as compact JSON text (athanor re-serializes them)."""
    msgs = []
    for m in messages:
        c = {"role": m["role"], "content": m.get("content")}
        if "reasoning_content" in m:
            c["reasoning"] = m["reasoning_content"]
        if "tool_calls" in m:
            c["tool_calls"] = [{"name": t["function"]["name"],
                                "arguments": json.dumps(t["function"]["arguments"], ensure_ascii=False, separators=(",", ":"))}
                               for t in m["tool_calls"]]
        if "tool_call_id" in m:
            c["tool_call_id"] = m["tool_call_id"]
        msgs.append(c)
    ctools = [{"name": t["function"]["name"], "description": t["function"].get("description"),
               "json_schema": json.dumps(t["function"]["parameters"], ensure_ascii=False, separators=(",", ":"))}
              for t in (tools or [])]
    return msgs, ctools


def generate(model_dir, tok):
    """Greedy decoding by the fp32 reference; margins; bf16 noise."""
    prompt = tok.apply_chat_template(GENERATE_PROMPT, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(prompt, add_special_tokens=False)
    m32 = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    dump.check_text_path(m32)
    m16 = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="eager").eval()
    dump.check_text_path(m16)
    dump.restore_f32(m16, model_dir)
    seq = list(ids)
    margins = []
    with torch.no_grad():
        for _ in range(GENERATE_STEPS):
            logits = m32(input_ids=torch.tensor([seq])).logits[0, -1].float()
            top = torch.topk(logits, 2)
            margins.append(float(top.values[0] - top.values[1]))
            seq.append(int(top.indices[0]))
        full = torch.tensor([seq[:-1]])
        l32 = m32(input_ids=full).logits[0, len(ids) - 1:].float()
        l16 = m16(input_ids=full).logits[0, len(ids) - 1:].float()
    gen = seq[len(ids):]
    noise = 0.0
    for k in range(GENERATE_STEPS):
        t32 = torch.topk(l32[k], 2)
        m32k = float(t32.values[0] - t32.values[1])
        m16k = float(l16[k][t32.indices[0]] - l16[k][t32.indices[1]])
        noise = max(noise, abs(m32k - m16k))
    return {"prompt_tokens": ids, "tokens": gen, "margins": margins, "noise": noise,
            "text": tok.decode(gen, skip_special_tokens=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "qwen35-0.8b"))
    args = ap.parse_args()
    dump.check_environment()
    torch.set_num_threads(1)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    tensors, corpus, templates = {}, [], []

    cases = list(CASES)
    for rel in REPO_FILES:
        with open(os.path.join(REPO, rel), encoding="utf-8") as f:
            cases.append(("file:" + rel, f.read()))
    readme = os.path.join(args.model_dir, "README.md")
    if os.path.exists(readme):
        with open(readme, encoding="utf-8") as f:
            cases.append(("file:model/README.md", f.read()))
    cases += all_code_points_cases() + decomposition_cases()

    for i, (name, text) in enumerate(cases):
        ids = tok.encode(text, add_special_tokens=False)
        back = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        tensors[f"corpus.{i}.text"] = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()
        tensors[f"corpus.{i}.ids"] = np.array(ids, dtype=np.int32)
        tensors[f"corpus.{i}.decoded"] = np.frombuffer(back.encode("utf-8"), dtype=np.uint8).copy()
        corpus.append({"name": name, "bytes": len(text.encode("utf-8")), "tokens": len(ids)})

    for i, (name, messages, tools, gen, think) in enumerate(TEMPLATES):
        rec = {"name": name, "generate": gen, "thinking": think}
        rec["messages"], rec["tools"] = c_view(messages, tools)
        try:
            text = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=gen,
                                           enable_thinking=think)
        except Exception as e:   # the template's raise_exception
            rec["error"] = str(e)
            templates.append(rec)
            continue
        tensors[f"template.{i}.text"] = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()
        tensors[f"template.{i}.ids"] = np.array(tok.encode(text, add_special_tokens=False), dtype=np.int32)
        templates.append(rec)

    gen = generate(args.model_dir, tok)

    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(args.out, "text.safetensors")
    save_file(tensors, data_path)
    manifest = {
        "format": 1,
        "about": "athanor text golden data: tests/golden/text.py, checked by tests/test_text.c",
        "reference": {
            "transformers": transformers.__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "python": platform.python_version(),
            "unicodedata": unicodedata.unidata_version,
            "tokenizer.json sha256": dump.sha256_file(os.path.join(args.model_dir, "tokenizer.json")),
            "chat_template.jinja sha256": dump.sha256_file(os.path.join(args.model_dir, "chat_template.jinja")),
        },
        "data": {"file": "text.safetensors", "sha256": dump.sha256_file(data_path), "tensors": len(tensors)},
        "corpus": corpus,
        "templates": templates,
        "generate": gen,
    }
    with open(os.path.join(args.out, "text.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    n_tok = sum(c["tokens"] for c in corpus)
    print(f"text.py: {len(corpus)} corpus texts ({n_tok} tokens), {len(templates)} templates "
          f"({sum('error' in t for t in templates)} errors), generate {len(gen['tokens'])} tokens: {gen['text']!r}")
    print(f"text.py: generate margins min {min(gen['margins']):.3f}, bf16 noise {gen['noise']:.3f}")


if __name__ == "__main__":
    main()
