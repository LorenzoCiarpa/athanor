"""Greedy generations of the reference implementation, for the step 8 gate.

For each of 50 chat prompts, 256 tokens of greedy decoding by the fp32
reference (the exact model) with the KV and recurrent cache, as a server
decodes, and the top-2 margin of every step. Then the bf16 reference (the
checkpoint's dtype, with the f32 tensors kept as stored, like dump.py) runs
the same tokens in one call: the largest change of the fp32 top-2 margin in
bf16 is the prompt's bf16 noise. Where the fp32 margin is within the noise,
rounding alone can flip the greedy choice, so a bf16 engine may part from
the fp32 sequence there and nowhere before (tests/test_engine.c,
generate_golden; tests/README.md, Engine).

Decoding does not stop at the end-of-turn token: the gate asks for 256
tokens, and both sides continue the same way.

Development only, about 20 minutes on 8 CPU threads; `make golden-generate`.
"""
import argparse
import json
import os
import platform
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dump  # noqa: E402

STEPS = 256


def u(text):
    return [{"role": "user", "content": text}]


def su(system, text):
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


# (name, messages, thinking). Topics, languages, formats and lengths vary so
# that the generations cover prose, lists, code, numbers and several scripts.
PROMPTS = [
    ("hash_table", u("Explain in detail how a hash table works."), False),
    ("photosynthesis", u("Describe photosynthesis step by step for a high school student."), False),
    ("story_lighthouse", u("Write a short story about a lighthouse keeper who finds a message in a bottle."), False),
    ("python_fib", u("Write a Python function that returns the n-th Fibonacci number, with tests."), False),
    ("c_linked_list", u("Implement a singly linked list in C with push, pop and reverse."), False),
    ("sql_join", u("Explain the difference between INNER JOIN, LEFT JOIN and FULL OUTER JOIN with examples."), False),
    ("rome_history", u("Summarize the history of the Roman Republic."), False),
    ("tcp_handshake", u("How does the TCP three-way handshake work, and why is it needed?"), False),
    ("recipe_risotto", u("Give me a detailed recipe for mushroom risotto."), False),
    ("math_series", u("Prove that the sum of the first n odd numbers is n squared."), False),
    ("it_vacanze", u("Scrivi una lettera a un amico per raccontargli le tue vacanze in Sicilia."), False),
    ("it_spiegazione", u("Spiegami in modo semplice come funziona una rete neurale."), False),
    ("fr_paris", u("Décris les monuments les plus importants de Paris."), False),
    ("de_energie", u("Erkläre die Vor- und Nachteile erneuerbarer Energien."), False),
    ("es_cuento", u("Escribe un cuento corto sobre un dragón que tenía miedo del fuego."), False),
    ("zh_tea", u("请介绍一下中国茶文化的历史。"), False),
    ("ja_seasons", u("日本の四季について説明してください。"), False),
    ("ru_space", u("Расскажи об истории освоения космоса."), False),
    ("pt_brasil", u("Fale sobre a culinária típica do Brasil."), False),
    ("ko_hangul", u("한글의 역사에 대해 설명해 주세요."), False),
    ("json_schema", u("Write a JSON schema for a library catalog with books, authors and loans."), False),
    ("regex", u("Explain this regular expression step by step: ^(?:[a-z0-9._%+-]+)@(?:[a-z0-9-]+\\.)+[a-z]{2,}$"), False),
    ("git_rebase", u("When should I use git rebase instead of git merge? Give concrete examples."), False),
    ("bash_script", u("Write a bash script that backs up a directory into a dated tar.gz archive and keeps the last 7."), False),
    ("rust_ownership", u("Explain ownership and borrowing in Rust with code examples."), False),
    ("haiku_series", u("Write ten haiku about the ocean."), False),
    ("poem_city", u("Write a poem about a city waking up in the morning."), False),
    ("dialogue", u("Write a dialogue between a teacher and a student about why the sky is blue."), False),
    ("list_tips", u("Give me 15 tips to improve my sleep."), False),
    ("compare_langs", u("Compare Python, Go and Rust for writing a web server."), False),
    ("physics_relativity", u("Explain special relativity without equations."), False),
    ("economics_inflation", u("What causes inflation, and how do central banks respond?"), False),
    ("biology_dna", u("How is DNA copied when a cell divides?"), False),
    ("chemistry_bonds", u("Explain ionic, covalent and metallic bonds."), False),
    ("table_planets", u("Make a markdown table of the planets of the solar system with their diameter and number of moons."), False),
    ("sys_pirate", su("You are a pirate. Answer every question in the voice of a pirate.", "How do I make a cup of coffee?"), False),
    ("sys_concise", su("You are a concise assistant. Use short sentences.", "Explain how vaccines work."), False),
    ("sys_tutor", su("You are a patient math tutor.", "I don't understand fractions. Can you help me?"), False),
    ("sys_it", su("Rispondi sempre in italiano.", "What are the main causes of climate change?"), False),
    ("multi_turn", [
        {"role": "user", "content": "I want to learn to play the guitar."},
        {"role": "assistant", "content": "That's great! Do you have a guitar already, and do you prefer acoustic or electric?"},
        {"role": "user", "content": "I have an acoustic one. Where should I start?"},
    ], False),
    ("multi_turn_code", [
        {"role": "user", "content": "Write a function that checks if a string is a palindrome."},
        {"role": "assistant", "content": "def is_palindrome(s):\n    return s == s[::-1]"},
        {"role": "user", "content": "Make it ignore spaces, punctuation and case, and add a docstring."},
    ], False),
    ("long_numbers", u("List the first 40 prime numbers and explain how you found them."), False),
    ("unicode_mix", u("Translate 'good morning, how are you?' into 12 languages, one per line."), False),
    ("think_puzzle", u("A farmer has 17 sheep and all but 9 run away. How many are left?"), True),
    ("think_math", u("What is the sum of the integers from 1 to 200 that are divisible by 3 or 5?"), True),
    ("think_code", u("Write a function that merges two sorted lists into one sorted list."), True),
    ("think_logic", u("If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?"), True),
    ("think_plan", u("Plan a three-day trip to Rome on a small budget."), True),
    ("think_it", u("Quanti giorni ci sono in tre anni, di cui uno bisestile?"), True),
    ("think_word", u("How many times does the letter r appear in the word strawberry?"), True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "qwen35-0.8b"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", type=int, default=0, help="first N prompts only (a quick check of the script)")
    args = ap.parse_args()
    dump.check_environment()
    torch.set_num_threads(args.threads)
    assert len(PROMPTS) == 50 and len({p[0] for p in PROMPTS}) == 50

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    m32 = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    dump.check_text_path(m32)
    m16 = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, attn_implementation="eager").eval()
    dump.check_text_path(m16)
    dump.restore_f32(m16, args.model_dir)

    records = []
    prompts = PROMPTS[: args.only] if args.only else PROMPTS
    t_start = time.time()
    for i, (name, messages, thinking) in enumerate(prompts):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        ids = tok.encode(text, add_special_tokens=False)
        tokens, seconds, margins = [], [], []
        with torch.no_grad():
            out = m32(input_ids=torch.tensor([ids]), use_cache=True)
            for _ in range(STEPS):
                logits = out.logits[0, -1].float()
                top = torch.topk(logits, 2)
                margins.append(float(top.values[0] - top.values[1]))
                t = int(top.indices[0])
                tokens.append(t)
                seconds.append(int(top.indices[1]))
                out = m32(input_ids=torch.tensor([[t]]), past_key_values=out.past_key_values, use_cache=True)
            # the same tokens through the bf16 reference, in one call
            full = torch.tensor([ids + tokens[:-1]])
            l16 = m16(input_ids=full).logits[0, len(ids) - 1:].float()
        noise = 0.0   # the largest change of the fp32 top-2 margin in the bf16 run
        for k in range(STEPS):
            m16k = float(l16[k][tokens[k]] - l16[k][seconds[k]])
            noise = max(noise, abs(margins[k] - m16k))
        records.append({"name": name, "thinking": thinking, "prompt": text, "prompt_tokens": ids, "tokens": tokens,
                        "margins": margins, "noise": noise})
        print(f"generate.py: {i + 1}/{len(prompts)} {name}: {len(ids)} + {STEPS} tokens, min margin "
              f"{min(margins):.3f}, noise {noise:.3f}, {time.time() - t_start:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "format": 1,
        "about": "athanor generation golden data: tests/golden/generate.py, checked by tests/test_engine.c",
        "reference": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "decoding": "greedy, fp32, eager attention, KV and recurrent cache; noise from the bf16 model "
                        "(f32 tensors as stored) on the same tokens in one call",
            "chat_template.jinja sha256": dump.sha256_file(os.path.join(args.model_dir, "chat_template.jinja")),
        },
        "steps": STEPS,
        "prompts": records,
    }
    with open(os.path.join(args.out, "generate.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False)
        f.write("\n")
    print(f"generate.py: {len(records)} prompts in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
