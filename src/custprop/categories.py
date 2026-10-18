"""Product categories from the item description.

The dataset has no product hierarchy, only free-text descriptions such as
"WHITE HANGING HEART T-LIGHT HOLDER". Keyword rules give each line a coarse category,
first match wins. They're a stand-in for a real product hierarchy and leave a fair share
of items as "other"; the share is reported in the cleaning notebook.
"""

from __future__ import annotations

import re

from pyspark.sql import Column
from pyspark.sql import functions as F

# Order matters: "CHRISTMAS MUG" is christmas, "LUNCH BAG" is bags.
RULES: list[tuple[str, tuple[str, ...]]] = [
    ("christmas", ("CHRISTMAS", "XMAS", "ADVENT", "SANTA", "REINDEER")),
    ("bags", ("BAG", "TOTE", "PURSE", "SHOPPER")),
    ("kitchen", ("MUG", "CUP", "PLATE", "BOWL", "TEA", "CAKE", "BAKING", "LUNCH BOX", "JAR",
                 "CUTLERY", "NAPKIN", "TRAY", "TIN", "SPOON", "EGG")),
    ("lighting", ("CANDLE", "T-LIGHT", "LIGHT", "LANTERN", "LAMP")),
    ("stationery", ("CARD", "WRAP", "NOTEBOOK", "PENCIL", "STICKER", "PAPER", "PEN ", "ENVELOPE")),
    ("decor", ("SIGN", "FRAME", "CUSHION", "CLOCK", "MIRROR", "HEART", "BUNTING", "DOORMAT", "GARLAND",
               "ORNAMENT", "DECORATION", "VASE")),
    ("toys", ("TOY", "GAME", "DOLL", "PLAYHOUSE", "JIGSAW", "SKIPPING ROPE", "SPACEBOY")),
]
CATEGORIES = [name for name, _ in RULES] + ["other"]


def _pattern(words: tuple[str, ...]) -> str:
    return "(" + "|".join(re.escape(w) for w in words) + ")"


def category_of(description: str | None) -> str:
    """Plain-Python version, for tests and pandas use."""
    text = (description or "").upper()
    for name, words in RULES:
        if re.search(_pattern(words), text):
            return name
    return "other"


def category_expr(description: Column) -> Column:
    text = F.upper(F.coalesce(description, F.lit("")))
    expr = F.lit("other")
    for name, words in reversed(RULES):  # built inside out so the first rule is checked first
        expr = F.when(text.rlike(_pattern(words)), F.lit(name)).otherwise(expr)
    return expr
