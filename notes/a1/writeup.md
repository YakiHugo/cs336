# Assignment 1 — written answers draft

Use this while reading `docs/handouts/cs336_assignment1_basics.md`.  
Final submission is a typeset `writeup.pdf`; this file is your working draft.

---

## §2.1 Problem (unicode1) — Understanding Unicode (1 point)

Status: **done** (concepts confirmed 2026-07-17; formal writeup optional)

**(a)** What Unicode character does `chr(0)` return?

> The NUL character (U+0000).

**(b)** How does this character’s string representation (`__repr__()`) differ from its printed representation?

> `repr` shows an escape like `'\x00'`; `print` typically emits nothing visible.

**(c)** What happens when this character occurs in text?

> It is a valid code point inside a Python `str`, but display/tools may treat it as a control character (invisible, truncate, or warn).

---

## §2.2 Problem (unicode2) — Unicode Encodings (3 points)

Status: **done** (concepts confirmed 2026-07-17; formal writeup optional)

**(a)** Reasons to prefer UTF-8 over UTF-16 / UTF-32 for tokenizer training?

> UTF-8 is variable-width, ASCII-compatible, and dominant on the web; UTF-16/32 waste space on ASCII-heavy text and complicate a small byte alphabet story less cleanly for byte-level BPE.

**(b)** Why is `decode_utf8_bytes_to_str_wrong` incorrect? Give a counterexample.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

> Counterexample: multi-byte UTF-8 such as `"牛".encode("utf-8")` (or `"こんにちは"`).  
> Why: it decodes each byte in isolation; continuation bytes are not valid standalone UTF-8 sequences.

**(c)** Give a two-byte sequence that does not decode to any Unicode character(s).

> e.g. `b"\x80\x80"` (orphaned continuation bytes) — not a valid UTF-8 encoding of any character. 

---

## Later sections

*(fill as you go — experiment questions, ablations, etc.)*
