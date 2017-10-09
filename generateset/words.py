raw_text = open('output.txt', encoding = 'utf-8').read()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

non_korean = [char for char in chars if ord(char) not in list(range(0x3131, 0x3164)) and ord(char) not in list(range(0xAC00, 0xD79E))]
korean = [char for char in chars if ord(char) in list(range(0x3131, 0x3164)) or ord(char) in list(range(0xAC00, 0xD79E))]