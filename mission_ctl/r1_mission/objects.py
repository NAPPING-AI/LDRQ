"""Object words -> a COCO class the detector knows, decided by a table, not by the model.

Stage E keeps stage C's split: the model only COPIES the words that name the object
("the red chair"), and this module decides what they mean. Three steps, each a reason
to refuse with `unknown_object`:

  1. provenance  every content word the model wrote must be in what the operator said
                 (normalized). A model that writes "chair" for "seat" has paraphrased,
                 and a paraphrase is exactly what the confirmation step cannot see.
  2. head        the noun phrase before the first preposition: "the cup on the table"
                 is the cup.
  3. table       the head must contain exactly one phrase from SYNONYMS. Two kinds
                 ("the chair or the sofa") or none ("the door", "the kitchen") refuse.

The table is conservative on purpose: a word is listed only if YOLOX-Nano's COCO class
covers what an operator means by it. "desk" is left out (COCO has none; desks come
back as "dining table" some of the time, which is not a thing to promise), so is "bag"
(three COCO classes), "glass" (cup or wine glass) and "me" (which person is the
speaker is not something a detector knows).

Stdlib only, Python 3.8.
"""
from __future__ import annotations

import re

SYNONYMS = {
    "person": ("person", "people", "man", "men", "woman", "women", "human", "guy", "lady",
               "someone", "somebody", "boy", "girl", "child", "kid"),
    "chair": ("chair", "office chair", "armchair", "seat", "stool"),
    "couch": ("couch", "sofa"),
    "bed": ("bed",),
    "dining table": ("table", "dining table"),
    "toilet": ("toilet",),
    "tv": ("tv", "television", "monitor", "screen", "display"),
    "laptop": ("laptop",),
    "mouse": ("computer mouse", "mouse"),
    "keyboard": ("keyboard",),
    "cell phone": ("phone", "cell phone", "cellphone", "mobile phone", "smartphone"),
    "book": ("book",),
    "clock": ("clock",),
    "vase": ("vase",),
    "bottle": ("bottle", "water bottle"),
    "cup": ("cup", "mug"),
    "bowl": ("bowl",),
    "potted plant": ("plant", "potted plant", "flower pot", "flowerpot"),
    "backpack": ("backpack", "rucksack"),
    "handbag": ("handbag", "purse"),
    "suitcase": ("suitcase", "luggage"),
    "umbrella": ("umbrella",),
    "sports ball": ("ball", "football", "soccer ball", "basketball"),
    "teddy bear": ("teddy bear", "teddy", "stuffed animal", "plush toy"),
    "refrigerator": ("refrigerator", "fridge"),
    "microwave": ("microwave",),
    "oven": ("oven",),
    "sink": ("sink",),
    "bench": ("bench",),
    "bicycle": ("bicycle", "bike"),
    "cat": ("cat", "kitten"),
    "dog": ("dog", "puppy"),
}

_DETERMINERS = {"the", "a", "an", "that", "this", "those", "these", "my", "your", "our",
                "its", "their", "his", "her", "one", "some", "any"}
_PREPOSITIONS = ("on", "in", "at", "by", "near", "next", "beside", "behind", "under",
                 "over", "with", "of", "from", "to", "in front", "close", "between",
                 "which", "that", "who", "where")
_WORD = re.compile(r"[a-z0-9]+")

# phrase -> class, longest phrases first so "dining table" beats "table" and
# "cell phone" beats "phone".
_PHRASES = sorted(((p, c) for c, ps in SYNONYMS.items() for p in ps),
                  key=lambda pc: -len(pc[0]))


def words(text):
    return _WORD.findall(str(text).lower())


def head_phrase(text):
    """Determiners dropped, cut at the first preposition or relative pronoun."""
    ws = words(text)
    while ws and ws[0] in _DETERMINERS:
        ws = ws[1:]
    for i, w in enumerate(ws):
        if i > 0 and (w in _PREPOSITIONS or " ".join(ws[i:i + 2]) in _PREPOSITIONS):
            ws = ws[:i]
            break
    return ws


def _plural(w):
    return (w + "s", w + "es", w[:-1] + "ies" if w.endswith("y") else w + "s")


def _has_phrase(ws, phrase):
    """`phrase` (one or more words) occurs in the word list, the last word possibly
    plural."""
    pw = phrase.split()
    n = len(pw)
    for i in range(len(ws) - n + 1):
        if ws[i:i + n - 1] == pw[:-1] and (ws[i + n - 1] == pw[-1]
                                          or ws[i + n - 1] in _plural(pw[-1])):
            return True
    return False


def resolve(text):
    """Object words -> (coco class or None, why). `why` is "" on success, else the
    reason in words for the refusal message."""
    hw = head_phrase(text)
    if not hw:
        return None, "no object named"
    found, used = [], []
    for phrase, cls in _PHRASES:
        if any(phrase in u for u in used):
            continue                      # "table" inside an already matched "dining table"
        if _has_phrase(hw, phrase):
            used.append(phrase)
            if cls not in found:
                found.append(cls)
    if not found:
        return None, "'{}' is not something I can recognize".format(" ".join(hw))
    if len(found) > 1:
        return None, "'{}' names more than one kind of thing ({})".format(
            " ".join(hw), ", ".join(found))
    return found[0], ""


def said(object_text, said_text):
    """Every content word of the model's object text is in what was said. Plural and
    singular count as the same word."""
    heard = set(words(said_text))
    for w in words(object_text):
        if w in _DETERMINERS:
            continue
        if w in heard:
            continue
        if any(p in heard for p in _plural(w)) or any(w == p for h in heard for p in _plural(h)):
            continue
        return False
    return True


def known_kinds():
    """The supported kinds, as an operator would say them, for messages."""
    return sorted(SYNONYMS)
