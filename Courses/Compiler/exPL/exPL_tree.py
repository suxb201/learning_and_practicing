import global_variable
import os


def error(s):
    print('ERROR:  ' + s)

    print()
    text.output()
    print()
    table.output()
    print()
    code.output()
    exit()


def test(s0, s1):
    if s0 != s1:
        error('test,' + s0 + ',' + s1)
    pass


def get_id(s, shape=0):
    global fk_id
    fk_id += 1
    fk_f.write('\t' + str(fk_id) + f' [color=Red,label="{s}",shape={"box" if shape == 0 else "ellipse"}]\n')
    return fk_id


def add_edge(id0, id1):
    fk_f.write('\t' + str(id0) + '->' + str(id1) + '\n')


# 处理单个元素 常数 变量
def factor(id_factor):
    sym = get_sym()
    # print(sym)
    if sym.isdigit():  # 是数字
        code.add('lit', 0, int(sym))  # 数字直接扔上去
        id_number = get_id('数字')
        add_edge(id_factor, id_number)
        add_edge(id_number, get_id(sym))
    else:  # 否则是标识符
        item = table.get(sym)
        id_identity = get_id('标识符')
        add_edge(id_factor, id_identity)
        add_edge(id_identity, get_id(sym))
        if item['kind'] == 'const':  # 是常量的话
            value = item['value']
            code.add('lit', 0, value)  # 常量的层数留空
        elif item['kind'] == 'var':
            the_level = item['level']
            the_addr = item['address']
            code.add('lod', level - the_level, the_addr)
        else:
            error("factor wrong")


# 处理乘法除法
def term(id_term):
    id_factor = get_id('因子')
    add_edge(id_term, id_factor)
    factor(id_factor)
    while get_next_sym() in ['*', '/']:
        sym = get_sym()
        id_mul_div = get_id('乘除运算符')
        add_edge(id_term, id_mul_div)
        add_edge(id_mul_div, get_id(sym))
        id_factor = get_id('因子')
        add_edge(id_term, id_factor)
        factor(id_factor)
        if sym == '*':
            code.add('opr', 0, 4)
        else:
            code.add('opr', 0, 5)


# 表达式解析
def expression(id_expression):
    # print('exp')
    if get_next_sym() in ['+', '-']:
        id_plus_sub = get_id('加减运算符')
        add_edge(id_expression, id_plus_sub)
        sym = get_sym()
        add_edge(id_plus_sub, get_id(sym))
        id_term = get_id('项')
        add_edge(id_expression, id_term)
        term(id_term)  # 处理 term
        if sym == '-':
            code.add('opr', '0', '1')
    else:
        id_term = get_id('项')
        add_edge(id_expression, id_term)
        term(id_term)

    while get_next_sym() in ['+', '-']:
        id_plus_sub = get_id('加减运算符')
        add_edge(id_expression, id_plus_sub)
        sym = get_sym()
        add_edge(id_plus_sub, get_id(sym))
        id_term = get_id('项')
        add_edge(id_expression, id_term)
        term(id_term)  # 处理 term
        if sym == '+':
            code.add('opr', 0, 2)  # + 2
        elif sym == '-':
            code.add('opr', 0, 3)  # - 3
        else:
            error('express error')


# 判断语句
def condition(id_condition):
    if get_next_sym() == 'odd':  # 一元运算符 奇偶判断
        add_edge(id_condition, get_id('odd'))
        id_expression = get_id('表达式')
        add_edge(id_condition, id_expression)
        get_sym()
        expression(id_expression)
        code.add('opr', 0, 6)  # 6 对应 odd
    else:
        id_expression = get_id('表达式')
        add_edge(id_condition, id_expression)
        expression(id_expression)
        sym = get_sym()
        id_relation = get_id('关系运算符')
        add_edge(id_condition, id_relation)
        add_edge(id_relation, get_id('sym'))
        id_expression = get_id('表达式')
        add_edge(id_condition, id_expression)
        expression(id_expression)
        if sym == '=':
            code.add('opr', 0, 8)
        elif sym == '#':
            code.add('opr', 0, 9)
        elif sym == '<':
            code.add('opr', 0, 10)
        elif sym == '<=':
            code.add('opr', 0, 13)
        elif sym == '>':
            code.add('opr', 0, 12)
        elif sym == '>=':
            code.add('opr', 0, 11)


# 从 begin 到 end （不包括 end 后面的符号）
def statement(id_statement):
    if get_next_sym() == 'begin':
        id_compound_statement = get_id('复合语句')
        add_edge(id_statement, id_compound_statement)
        add_edge(id_compound_statement, get_id('begin'))
        test(get_sym(), 'begin')
        id_statement0 = get_id('语句')
        add_edge(id_compound_statement, id_statement0)
        statement(id_statement0)
        while get_next_sym() == ';':
            test(get_sym(), ';')
            add_edge(id_compound_statement, get_id(';'))
            id_statement0 = get_id('语句')
            add_edge(id_compound_statement, id_statement0)
            statement(id_statement0)
        test(get_sym(), 'end')
        add_edge(id_compound_statement, get_id('end'))
    elif get_next_sym() == 'if':
        id_condition_statement = get_id('条件语句')
        add_edge(id_statement, id_condition_statement)
        add_edge(id_condition_statement, get_id('if'))
        test(get_sym(), 'if')
        id_condition = get_id('条件')
        add_edge(id_condition_statement, id_condition)
        condition(id_condition)
        addr = code.add('jpc', 0, 0)
        test(get_sym(), 'then')
        add_edge(id_condition_statement, get_id('then'))
        id_statement0 = get_id('语句')
        add_edge(id_condition_statement, id_statement0)
        statement(id_statement0)
        code.set(addr, code.length())
    elif get_next_sym() == 'while':
        id_while_statement = get_id('当型循环语句')
        add_edge(id_statement, id_while_statement)
        add_edge(id_while_statement, get_id('while'))
        test(get_sym(), 'while')
        addr0 = code.length()
        id_condition = get_id('条件')
        add_edge(id_while_statement, id_condition)
        condition(id_condition)
        addr1 = code.add('jpc', 0, '-')
        add_edge(id_while_statement, get_id('do'))
        test(get_sym(), 'do')
        id_statement0 = get_id('语句')
        add_edge(id_while_statement, id_statement0)
        statement(id_statement0)
        code.add('jmp', 0, addr0)
        code.set(addr1, code.length())
    elif get_next_sym() == 'call':
        id_call_statement = get_id('过程调用语句')
        add_edge(id_statement, id_call_statement)
        add_edge(id_call_statement, get_id('call'))
        test(get_sym(), 'call')
        name = get_sym()
        id_identity = get_id('标识符')
        add_edge(id_call_statement, id_identity)
        add_edge(id_identity, get_id(name))
        code.add(
            'cal',
            level - table.get(name)['level'],
            table.get(name)['address']
        )
    elif get_next_sym() == 'write':
        id_write_statement = get_id('写语句')
        add_edge(id_statement, id_write_statement)
        add_edge(id_write_statement, get_id('('))
        test(get_sym(), 'write')
        test(get_sym(), '(')
        while True:
            id_expression = get_id('表达式')
            add_edge(id_write_statement, id_expression)
            expression(id_expression)
            code.add('opr', 0, 14)  # write
            code.add('opr', 0, 15)  # 换行

            if get_next_sym() != ',':
                break
            add_edge(id_write_statement, get_id(','))
            test(get_sym(), ',')
        test(get_sym(), ')')
        add_edge(id_write_statement, get_id(')'))
    elif get_next_sym() == 'read':
        id_read_statement = get_id('读语句')
        add_edge(id_statement, id_read_statement)
        add_edge(id_read_statement, get_id('('))
        test(get_sym(), 'read')
        test(get_sym(), '(')
        while True:
            name = get_sym()
            id_identity = get_id('标识符')
            add_edge(id_read_statement, id_identity)
            add_edge(id_identity, get_id(name))
            code.add('opr', 0, 16)  # read bnhr55
            code.add(  # sto
                'sto',
                level - table.get(name)['level'],
                table.get(name)['address']
            )
            if get_next_sym() != ',':
                break
            add_edge(id_read_statement, get_id(','))
            test(get_sym(), ',')
        add_edge(id_read_statement, get_id(')'))
        test(get_sym(), ')')
    elif table.is_ident(get_next_sym()):
        id_assign_statement = get_id('赋值语句')
        id_identity = get_id('标识符')
        add_edge(id_statement, id_assign_statement)
        add_edge(id_assign_statement, id_identity)
        name = get_sym()
        add_edge(id_identity, get_id(name))
        add_edge(id_assign_statement, get_id(':='))
        test(get_sym(), ':=')
        id_expression = get_id('表达式')
        add_edge(id_assign_statement, id_expression)
        expression(id_expression)  # 解析语句
        # test(get_sym(), ';')  # 解析完成后应该是分号
        code.add(
            'sto',
            level - table.get(name)['level'],
            table.get(name)['address']
        )
    else:
        add_edge(id_statement, get_id('空'))


def block(id_sub_program):
    global level
    relative_addr = 2  # 相对地址，从 3 开始  0 1 2 被占用
    jmp_addr = code.add('jmp', 0, '-')  # 上来先塞上一个 jmp

    while get_next_sym() in ['const', 'var', 'procedure']:  # 是声明
        sym = get_sym()
        if sym == 'const':
            id_const_description = get_id("常量说明部分")
            add_edge(id_sub_program, id_const_description)
            add_edge(id_const_description, get_id('const'))
            id_const_definition = get_id("常量定义")
            add_edge(id_const_description, id_const_definition)
            while True:
                name = get_sym()
                test(get_sym(), '=')
                value = get_sym()
                value = int(value)
                table.add('const', name, value, '-', '-')
                id_identity = get_id('标识符')
                id_equal = get_id('=')
                id_number = get_id('无符号整数')
                add_edge(id_const_definition, id_identity)
                add_edge(id_const_definition, id_equal)
                add_edge(id_const_definition, id_number)
                add_edge(id_identity, get_id(name))
                add_edge(id_number, get_id(value))
                if get_sym() == ';':
                    break
                add_edge(id_const_definition, get_id(','))
            add_edge(id_const_description, get_id(';'))
        elif sym == 'var':
            id_variable_description = get_id('变量说明部分')
            add_edge(id_sub_program, id_variable_description)
            add_edge(id_variable_description, get_id('var'))
            while True:
                name = get_sym()
                relative_addr += 1
                table.add('var', name, '-', level, relative_addr)
                id_identity = get_id('标识符')
                add_edge(id_variable_description, id_identity)
                add_edge(id_identity, get_id(name))
                if get_sym() == ';':
                    break
                add_edge(id_variable_description, get_id(','))
            add_edge(id_variable_description, get_id(';'))
        elif sym == 'procedure':
            name = get_sym()
            test(get_sym(), ';')
            table.add('procedure', name, '-', level, code.length())
            level += 1
            id_proc_description = get_id("过程说明部分")
            id_proc_header = get_id("过程首部")
            id_sub_program0 = get_id("分程序")
            add_edge(id_sub_program, id_proc_description)
            add_edge(id_proc_description, id_proc_header)
            add_edge(id_proc_description, id_sub_program0)
            add_edge(id_proc_description, get_id(";"))
            add_edge(id_proc_header, get_id('procedure'))
            id_identity = get_id('标识符')
            add_edge(id_proc_header, id_identity)
            add_edge(id_identity, get_id(name))
            add_edge(id_proc_header, get_id(";"))
            block(id_sub_program0)
            level -= 1
            test(get_sym(), ';')

    code.set(jmp_addr, code.length())  # 反填跳转地址
    code.add('int', 0, relative_addr + 1)
    id_statement = get_id('语句')
    add_edge(id_sub_program, id_statement)
    statement(id_statement)
    code.add('opr', 0, 0)  # return


if __name__ == '__main__':
    fk_f = open('Syntax-Tree.dot', 'w', encoding='utf-8')  # 文件写
    fk_f.write('digraph hierarchy { \n'
               '\trankdir=LR;  //Rank Direction Left to Right \n'
               '\tedge [color=blue] \n'
               '\tnode [ fontname="YaHei Consolas Hybrid" size="20,20"]\n')

    text = global_variable.Text('./学长_in.txt')  # 读入的文本
    get_sym = text.get_sym
    get_next_sym = text.get_next_sym
    table = global_variable.Table()  # 符号表
    code = global_variable.Code()  # 目标代码表

    level = 0
    fk_id = 0
    root = get_id("root")
    pro = get_id("程序")
    sub = get_id("分程序")
    add_edge(root, pro)
    add_edge(pro, sub)
    block(sub)
    test(get_sym(), '.')  # 吞掉 '.'

    fk_f.write('}\n')
    fk_f.close()
    os.system('dot Syntax-Tree.dot -Tpng -o Syntax-Tree.png')

    # print()
    # text.output()
    # print()
    # table.output()
    # print()
    # code.output()



