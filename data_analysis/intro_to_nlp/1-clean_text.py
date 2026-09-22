#!/usr/bin/env python3
"""
Clean text data from PII, emoji,
normalise text messages.
"""
import re
import emoji


_DATASET_PLACEHOLDER_MAP = {
    '<#>':       '<NUM>',
    '<decimal>': '<NUM>',
    '<time>':    '<TIME>',
    '<url>':     '<URL>',
    '<email>':   '<EMAIL>',
}
_URL_RE = re.compile(
    r'https?://\S+|www\.\S+'
)
_PHONE_RE = re.compile(
    r'\+?\d[\d\s\-]{6,}\d'
)
_NUMBER_RE = re.compile(
    r'(?:£|\$|€)\d+(?:[.,]\d+)*|(?<!<)\b\d+(?:[.,]\d+)*\b'
)
_REPEAT_BANG_RE = re.compile(r'!{2,}')
_REPEAT_QMARK_RE = re.compile(r'\?{2,}')
_WHITESPACE_RE = re.compile(r'\s+')


def normalize_unicode_punct(text):
    """
    Replace curly quotes, dashes, ellipses,
    etc. with ASCII equivalents.
    """
    replacements = {
        # '[\u2018\u2019\u201a\u201b]': "'",
        # '[\u201c\u201d\u201e\u201f]': '"',
        '[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]': '-',
        '\u2026': '...'
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    return text


def clean_text(
    text,
    replace_num=True,
    replace_url=True,
    emoji_action="replace"
):
    """
    Lowercase, normalize, and mask
    variable content in text messages.
    """
    if text is None:
        return ''
    text = text.lower().strip()

    for placeholder, replacement in _DATASET_PLACEHOLDER_MAP.items():
        text = text.replace(placeholder, replacement)

    text = normalize_unicode_punct(text)

    if replace_url:
        text = _URL_RE.sub('<URL>', text)

    if replace_num:
        # phone-like runs first, or
        # the digit pass below eats them
        text = _PHONE_RE.sub('<NUM>', text)
        text = _NUMBER_RE.sub('<NUM>', text)

    if emoji_action == 'replace':
        text = emoji.replace_emoji(
            text, replace='<EMO>'
        )
    elif emoji_action == 'remove':
        text = emoji.replace_emoji(text, replace=' ')

    text = _REPEAT_BANG_RE.sub('!', text)
    text = _REPEAT_QMARK_RE.sub('?', text)

    return _WHITESPACE_RE.sub(' ', text).strip()
