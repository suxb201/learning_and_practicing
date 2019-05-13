# class TableNode:  # 符号表
#     def __init__(self):
#         self.kind = 0
#         self.name = 0
#         self.value = 0
#         self.level = 0
#         self.address = 0
#
#     @staticmethod
#     def output(self):
#         print(
#             self.kind.ljust(7),
#             self.name.ljust(13),
#             self.value.ljust(7),
#             self.level.ljust(7),
#             self.address.ljust(7)
#         )
#
# class CodeNode:
#     def __init__(self):  # 对应 f、l、a
#         self.F = 0
#         self.L = 0
#         self.A = 0
#
#     @staticmethod
#     def output(self):
#         print(
#             self.F.ljust(7),
#             self.L.ljust(7),
#             self.A.ljust(7)
#         )


class Text:
    def __init__(self, file_name):
        self.s = str()
        with open(file_name) as f:
            self.s = f.read()
        chs = [
            '+', '-', '*',
            '/', '(', ')',
            '<', '>', '=',
            ':', ',', '.',
            ';', '#'
        ]
        for ch in chs:
            self.s = self.s.replace(ch, ' ' + ch + ' ')
        self.s = self.s.split()
        self.s = ' '.join(self.s)
        self.s = self.s.replace(': =', ':=')
        self.s = self.s.replace('< =', ':=')
        self.s = self.s.replace('> =', ':=')
        self.s = self.s.split()
        self.now_pointer = 0

    def get_sym(self):
        self.now_pointer = self.now_pointer + 1
        # print(self.s[self.now_pointer - 1])
        return self.s[self.now_pointer - 1]

    def get_next_sym(self):
        return self.s[self.now_pointer]

    def output(self):
        print(self.s)


class Table:
    def __init__(self):
        self.table = list()

    def add(self, kind, name, value, level, address):
        self.table.append([kind, name, value, level, address])

    # def set(self, name, value):
    #     for item in self.table:
    #         if item[1] == name:
    #             item[2] = value
    #             return
    def get(self, name):
        for item in self.table:
            if item[1] == name:
                return {
                    'kind': item[0],
                    'name': item[1],
                    'value': item[2],
                    'level': item[3],
                    'address': item[4]
                }

    def is_ident(self, name):
        for item in self.table:
            if item[1] == name:
                return True
        return False

    def output(self):
        print('[table]')
        print(f'| {"index":<6}'
              f'| {"kind":<11}'
              f'| {"name":<13}'
              f'| {"value":<6}'
              f'| {"level":<6}'
              f'| {"address":<8}')
        for index, item in enumerate(self.table):
            print(f'| {index:<6}'
                  f'| {item[0]:<11}'
                  f'| {item[1]:<13}'
                  f'| {item[2]:<6}'
                  f'| {item[3]:<6}'
                  f'| {item[4]:<8}')


class Code:
    def __init__(self):
        self.code = list()

    def length(self):
        return len(self.code)

    def add(self, f, l, a):
        self.code.append([f, l, a])
        return len(self.code) - 1

    def set(self, index, a):
        self.code[index][2] = a

    # def pop(self):
    #     self.code.pop()

    def output(self):
        print('[code]')
        print(f'| {"index":<6}'
              f'| {"f":<4}'
              f'| {"l":<4}'
              f'| {"a":<4}')
        for index, item in enumerate(self.code):
            print(f'| {index:<6}'
                  f'| {item[0]:<4}'
                  f'| {item[1]:<4}'
                  f'| {item[2]:<4}')
