THE_SYM = {
    "const": 1,
    "var": 2,
    "procedure": 3,
    ":=": 4,
    "begin": 5,
    "end": 6,
    "if": 7,
    "then": 8,
    "call": 9,
    "while": 10,
    "do": 11,
    "read": 12,
    "write": 13,
    "+": 14,
    "-": 15,
    "*": 16,
    "/": 17,
    ">": 18,
    "<": 19,
    "=": 20,
    "#": 21,
    ";": 22,
    "number": 23,
    ".": 24,
    ",": 25,
    "(": 26,
    ")": 27
}

globalSym = []  # 符号表
globalId = -1  # 最大数字
globalNum = []  # 数字和变量表


def get_symbol(text):
    global globalId
    global globalSym
    global globalNum

    for c in THE_SYM:
        text = text.replace(c, ' ' + c + ' ')
    text = text.split()
    for i in range(1, len(text)):
        if text[i - 1] in ['<', '>', ':'] and text[i] == '=':
            text[i] = text[i - 1] + text[i]
            text[i - 1] = ''
    while '' in text:
        text.remove('')
    for i in text:
        if i in THE_SYM:  # 如果是符号表里的
            globalSym.append([i, THE_SYM[i], None])
        else:
            if i.isdigit():  # 如果不是，说明是变量
                globalNum.append([i, int(i)])
            else:
                globalNum.append([i, None])
            globalId = globalId + 1
            globalSym.append([i, THE_SYM["number"], globalId])

    print("符号".ljust(8), "类别".ljust(4), "地址".ljust(5))
    for i in globalSym:
        print(i[0].ljust(9), str(i[1]).ljust(6), str(i[2]).ljust(5))

    print("-------------------")
    for i in globalNum:
        print(i[0].ljust(9), str(i[1]).ljust(6))
    return text


if __name__ == '__main__':
    with open("./学长_in.txt", "r") as f:
        originText = f.read()
    sym = get_symbol(originText)
    print(sym)
