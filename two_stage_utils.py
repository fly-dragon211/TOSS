import re


def sliding_window(token_list, window_size, step):
    for y in range(0, len(token_list) - 1, step):
        yield (y, token_list[y:y + window_size])