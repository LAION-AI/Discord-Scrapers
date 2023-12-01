from typing import Tuple


start_quotes = [
    '"',
    "“",
    "«",
    "„",
]

end_quotes = [
    '"',
    "”",
    "»",
    "“",
]


def starts_with_quotes(string: str) -> bool:
    if len(string) == 0:
        return False
    return string[0] in start_quotes


def get_start_end_quotes(string: str) -> Tuple[int, int]:
    first_quote_index = -1
    last_quote_index = -1

    for i, char in enumerate(string):
        if first_quote_index == -1 and char in start_quotes:
            first_quote_index = i
        elif first_quote_index != -1 and char in end_quotes:
            last_quote_index = i

    return (first_quote_index, last_quote_index)


def has_code_block(string: str) -> bool:
    # Find the first occurrence of ```
    first_code_block_index = string.find("```")
    if first_code_block_index == -1:
        return False
    # Find the second occurrence of ```
    second_code_block_index = string.find("```", first_code_block_index + 3)
    if second_code_block_index == -1:
        return False
    return True


def get_code_block(string: str) -> str:
    first_code_block_index = string.find("```")
    second_code_block_index = string.find("```", first_code_block_index + 3)

    return string[first_code_block_index + 3 : second_code_block_index]
