import numpy as np

DATA_SETS = [
    ['Hello?', 'Hello.'],
    ['Who are you?', 'Jack.'],
    ['How old are you?', '25th.'],
    ['Nice to meet you.', 'Too.'],
    ['What`s wrong with you?', 'Fine.'],
    ['Bye bye.', 'Bye.'],
]


def get_data():
    words = set()
    inputs, outputs = [], []
    for data in DATA_SETS:
        inputs.append(data[0])
        outputs.append(data[1])

    return inputs, outputs
