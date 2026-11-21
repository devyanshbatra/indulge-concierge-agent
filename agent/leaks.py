"""Finds private member details in text before it can reach a vendor.

It checks every member on file, not only the one on the ticket, and looks for
each detail in many shapes: phone numbers with any spacing or spelled out as
words, first or last names on their own, address fragments, emails, card digits
and expiry dates. The house's own phone and email are allowed.
"""
from __future__ import annotations

import re

_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_PHONE_LIKE = re.compile(r"(?<![\w/])\+?\d[\d\s().\-]{6,}\d(?![\w/])")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_EXPIRY = re.compile(r"\b(?:0[1-9]|1[0-2])\s?/\s?(?:20\d{2}|\d{2})\b")
_POSTCODE = re.compile(r"(?<!\d)\d{6}(?!\d)")
_STREET = re.compile(r"\b\d{1,4}[A-Za-z]?,?\s+(?:[A-Z][a-z]+\s+){1,3}(?:Road|Rd|Street|St|Lane|Marg|Nagar|Avenue|Colony)\b")
_FLAT = re.compile(r"\b(?:flat|apt|apartment)\s*(?:no\.?\s*)?\d+", re.I)
_ADDRESS_STOPWORDS = {"road", "flat", "hill", "west", "east", "north", "south", "cross", "street", "lane",
                      "mumbai", "delhi", "bengaluru", "bangalore", "new", "palace"}


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _words_re(words) -> re.Pattern:
    """Whole-word, case-insensitive match for any of `words` (never matches if there are none)."""
    if not words:
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(" + "|".join(map(re.escape, sorted(words, key=len, reverse=True))) + r")\b", re.I)


class LeakScanner:
    def __init__(self, members, house: dict):
        self._allowed_digits = {_digits(house["house_phone"])}
        self._allowed_emails = {house["house_email"].lower()}
        names: dict[str, str] = {}
        self._phones: set[str] = set()
        address_words: set[str] = set()
        self._card_last4: set[str] = set()
        for member in members:
            for full_name in [member["name"], *(f["name"] for f in member.get("family", []))]:
                for part in full_name.split():
                    if len(part) >= 3:
                        names[part.lower()] = part
            if member.get("phone"):
                self._phones.add(_digits(member["phone"])[-10:])
            for word in re.findall(r"[A-Za-z]{4,}", member.get("address", "")):
                if word.lower() not in _ADDRESS_STOPWORDS:
                    address_words.add(word)
            if member.get("card", {}).get("last4"):
                self._card_last4.add(member["card"]["last4"])
        self._names = names
        self._name_re = _words_re(names)
        self._address_re = _words_re(address_words)

    def scan(self, text: str) -> list[str]:
        """Plain-words list of the private details found in `text` (empty means clean)."""
        found: list[str] = []

        def add(label: str) -> None:
            if label not in found:
                found.append(label)

        for m in self._name_re.finditer(text):
            add(f"name ({self._names[m.group(1).lower()]})")

        for m in _PHONE_LIKE.finditer(text):
            raw = m.group(0).strip()
            digits = _digits(raw)
            if _ISO_DATE.match(raw) or len(digits) < 8 or digits in self._allowed_digits:
                continue
            add("card number" if len(digits) >= 13 else "phone number")
        all_digits = _digits(text)
        spoken = self._spoken_runs(text)
        if any(p in all_digits or any(p in run for run in spoken) for p in self._phones):
            add("phone number")
        elif any(len(run) >= 8 for run in spoken):
            add("phone number")

        for m in _EMAIL.finditer(text):
            if m.group(0).lower() not in self._allowed_emails:
                add("email address")

        for m in self._address_re.finditer(text):
            add(f"address detail ({m.group(1)})")
        if _POSTCODE.search(text):
            add("postcode")
        if _STREET.search(text) or _FLAT.search(text):
            add("street address")

        for last4 in self._card_last4:
            if re.search(rf"(?<!\d){last4}(?!\d)", text):
                add("card digits")
        if _EXPIRY.search(text):
            add("card expiry date")
        return found

    @staticmethod
    def _spoken_runs(text: str) -> list[str]:
        """Runs of digits that include at least one spelled-out digit ("nine 8 two zero ...")."""
        runs, run, has_word = [], "", False
        for token in re.findall(r"[a-z]+|\d+", text.lower()) + ["."]:
            digit = _DIGIT_WORDS.get(token, token if token.isdigit() else None)
            if digit is not None:
                run, has_word = run + digit, has_word or not token.isdigit()
                continue
            if has_word and run:
                runs.append(run)
            run, has_word = "", False
        return runs
