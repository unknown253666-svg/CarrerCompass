import re

def preprocess_text(text: str) -> str:
    """
    1. Lower-case
    2. Insert space before any capital letter that follows a lower-case letter
       (fixes merged words from PDF copy-paste)
    3. Collapse multiple whitespaces
    """
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)   # camelCase → camel case
    text = re.sub(r'\s+', ' ', text)                   # collapse spaces
    return text.lower().strip()