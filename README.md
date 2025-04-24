# toc
1. Design a DFA which will accept all the strings containing even number of 0's and even number of 1’s over an alphabet {0, 1} and write a program to implement the DFA.

=> def dfa_accepts(string):
    state = "S00"
    transitions = {
        "S00": {"0": "S10", "1": "S01"},
        "S01": {"0": "S11", "1": "S00"},
        "S10": {"0": "S00", "1": "S11"},
        "S11": {"0": "S01", "1": "S10"}
    }
    for char in string:
        state = transitions[state][char]
    return state == "S00"

string = input("Enter a binary string: ")
print("Accepted" if dfa_accepts(string) else "Rejected")
____________________________________________________________________________________________________________________________________________________________________________________________

2. Design a DFA which will accept all the strings containing mod 3 of 0’s over an alphabet {0, 1} and write a program to implement the DFA.

=> def dfa_mod3_zeros(string):
    state = "S0"
    transitions = {
        "S0": {"0": "S1", "1": "S0"},
        "S1": {"0": "S2", "1": "S1"},
        "S2": {"0": "S0", "1": "S2"}
    }
    for char in string:
        state = transitions[state][char]
    return state == "S0"

string = input("Enter a binary string: ")
print("Accepted" if dfa_mod3_zeros(string) else "Rejected")
____________________________________________________________________________________________________________________________________________________________________________________________

3. Construct a regular expression. And Converting FA to Regular Expressions.

=> import re

def regex_accepts(string):
    pattern = r"^(1* (01*01*01*)*)*$"
    return re.fullmatch(pattern, string) is not None

string = input("Enter a binary string: ")
print("Accepted" if regex_accepts(string) else "Rejected")
____________________________________________________________________________________________________________________________________________________________________________________________

4. Write a code to convert Ambiguous to Unambiguous CFG.

=> def eliminate_ambiguity():
    print("Original Ambiguous Grammar:")
    print("E -> E + E | E * E | ( E ) | id")
    
    print("\nUnambiguous Grammar:")
    print("E -> TE'")
    print("E' -> +TE' | ε")
    print("T -> FT'")
    print("T' -> *FT' | ε")
    print("F -> (E) | id")

eliminate_ambiguity()

____________________________________________________________________________________________________________________________________________________________________________________________

5. Write a code for simplification of Grammar.

=> from collections import defaultdict

def remove_useless_symbols(grammar, start_symbol):
    reachable = set()
    def dfs(symbol):
        if symbol in reachable:
            return
        reachable.add(symbol)
        for production in grammar.get(symbol, []):
            for sym in production:
                if sym.isupper():
                    dfs(sym)
    dfs(start_symbol)
    grammar = {k: v for k, v in grammar.items() if k in reachable}
    return grammar

def remove_null_productions(grammar):
    nullable = {k for k, v in grammar.items() if 'ε' in v}
    new_grammar = defaultdict(set)
    for lhs, productions in grammar.items():
        for production in productions:
            new_grammar[lhs].update(generate_non_nullable(production, nullable))
    return new_grammar

def generate_non_nullable(production, nullable):
    results = {''}
    for sym in production:
        if sym in nullable:
            results.update({r + sym for r in results})
        else:
            results = {r + sym for r in results}
    return results - {''}

def simplify_grammar(grammar, start_symbol):
    grammar = remove_useless_symbols(grammar, start_symbol)
    grammar = remove_null_productions(grammar)
    return grammar

grammar = {
    'S': {'AB', 'BC'},
    'A': {'aA', 'ε'},
    'B': {'b'},
    'C': {'cC', 'ε'}
}

simplified_grammar = simplify_grammar(grammar, 'S')
print("Simplified Grammar:")
for k, v in simplified_grammar.items():
    print(f"{k} -> {' | '.join(v)}")
____________________________________________________________________________________________________________________________________________________________________________________________

6. Design a lexical analyzer for given language and the lexical analyzer should ignore redundant spaces, tabs and new lines.

=> import re

def lexical_analyzer(code):
    tokens = []
    token_pattern = r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[+\-*/=(){};,]"
    code = re.sub(r"[\t\n ]+", " ", code)  # Remove redundant spaces, tabs, and new lines
    for match in re.finditer(token_pattern, code):
        tokens.append(match.group())
    return tokens

code = """
    int main() {
        int a = 10;
        int b = 20;
        int sum = a + b;
    }
"""

tokens = lexical_analyzer(code)
print("Tokens:", tokens)
____________________________________________________________________________________________________________________________________________________________________________________________

7.Implement Push Down Automata.

=> class PDA:
    def __init__(self):
        self.stack = []
        self.transitions = {
            ('q0', 'a', ''): ('q0', 'A'),
            ('q0', 'a', 'A'): ('q0', 'AA'),
            ('q0', 'b', 'A'): ('q1', ''),
            ('q1', 'b', 'A'): ('q1', ''),
            ('q1', '', ''): ('qf', '')
        }
        self.start_state = 'q0'
        self.final_state = 'qf'

    def process_input(self, input_string):
        state = self.start_state
        input_string += ' '
        for symbol in input_string:
            stack_top = self.stack[-1] if self.stack else ''
            if (state, symbol, stack_top) in self.transitions:
                state, stack_action = self.transitions[(state, symbol, stack_top)]
                if stack_top:
                    self.stack.pop()
                if stack_action:
                    self.stack.extend(reversed(stack_action))
            else:
                return False
        return state == self.final_state and not self.stack

pda = PDA()
string = input("Enter a string of a's followed by b's: ")
print("Accepted" if pda.process_input(string) else "Rejected")
____________________________________________________________________________________________________________________________________________________________________________________________

8.Converting PDA to CFG.

=> class CFG:
    def __init__(self):
        self.productions = {
            'S': ['aSb', '']
        }

    def generate(self, s):
        if s == "":
            return ""
        if s == "aSb":
            return "a" + self.generate("S") + "b"
        return s

cfg = CFG()
string = input("Enter a string with equal number of a's and b's: ")
print("Accepted" if string in cfg.generate('aSb'* (len(string)//2)) else "Rejected")
____________________________________________________________________________________________________________________________________________________________________________________________

9. Converting CFG to PDA.

=> class PDA:
    def __init__(self):
        self.stack = []
        self.transitions = {
            ('q0', 'a', ''): ('q0', 'A'),
            ('q0', 'a', 'A'): ('q0', 'AA'),
            ('q0', '', ''): ('q1', ''),
            ('q1', 'b', 'A'): ('q1', ''),
            ('q1', '', ''): ('qf', '')
        }
        self.start_state = 'q0'
        self.final_state = 'qf'

    def process_input(self, input_string):
        state = self.start_state
        input_string += ' '
        for symbol in input_string:
            stack_top = self.stack[-1] if self.stack else ''
            if (state, symbol, stack_top) in self.transitions:
                state, stack_action = self.transitions[(state, symbol, stack_top)]
                if stack_top:
                    self.stack.pop()
                if stack_action:
                    self.stack.extend(reversed(stack_action))
            else:
                return False
        return state == self.final_state and not self.stack

pda = PDA()
string = input("Enter a string with equal number of a's and b's: ")
print("Accepted" if pda.process_input(string) else "Rejected")
____________________________________________________________________________________________________________________________________________________________________________________________

10. Write an algorithm and program on Recursive Descent parser.

=>	class RecursiveDescentParser:
    def __init__(self, input_string):
        self.input = input_string.replace(" ", "") + "$"
        self.index = 0

    def match(self, char):
        if self.input[self.index] == char:
            self.index += 1
            return True
        return False

    def S(self):
        if self.A():
            if self.match('$'):
                return True
        return False

    def A(self):
        if self.match('a'):
            if self.A():
                return self.match('b')
            return False
        return True

    def parse(self):
        return self.S()

string = input("Enter a string with equal number of a's and b's: ")
parser = RecursiveDescentParser(string)
print("Accepted" if parser.parse() else "Rejected")
____________________________________________________________________________________________________________________________________________________________________________________________

11. Write an algorithm and program to compute FIRST and FOLLOW function.

=> from collections import defaultdict

def compute_first(grammar):
    first = defaultdict(set)
    
    def first_of(symbol):
        if symbol in first:
            return first[symbol]
        if not symbol.isupper():
            return {symbol}
        result = set()
        for production in grammar.get(symbol, []):
            for sym in production:
                sym_first = first_of(sym)
                result.update(sym_first - {'ε'})
                if 'ε' not in sym_first:
                    break
            else:
                result.add('ε')
        first[symbol] = result
        return result
    
    for non_terminal in grammar:
        first_of(non_terminal)
    return first

def compute_follow(grammar, start_symbol):
    follow = defaultdict(set, {start_symbol: {'$'}})
    first = compute_first(grammar)
    
    def follow_of(non_terminal):
        for lhs, productions in grammar.items():
            for production in productions:
                for i, symbol in enumerate(production):
                    if symbol == non_terminal:
                        next_symbols = production[i + 1:]
                        if next_symbols:
                            first_next = set()
                            for sym in next_symbols:
                                first_next.update(first[sym] - {'ε'})
                                if 'ε' not in first[sym]:
                                    break
                            else:
                                follow[non_terminal].update(follow[lhs])
                            follow[non_terminal].update(first_next)
                        else:
                            follow[non_terminal].update(follow[lhs])
    
    for _ in range(len(grammar)):
        for non_terminal in grammar:
            follow_of(non_terminal)
    return follow

grammar = {
    'E': ['TR'],
    'R': ['+TR', 'ε'],
    'T': ['FY'],
    'Y': ['*FY', 'ε'],
    'F': ['(E)', 'id']
}

first = compute_first(grammar)
follow = compute_follow(grammar, 'E')

print("FIRST:")
for k, v in first.items():
    print(f"{k}: {v}")

print("FOLLOW:")
for k, v in follow.items():
    print(f"{k}: {v}")
____________________________________________________________________________________________________________________________________________________________________________________________

12. Develop an operator precedence parser for a given language.

=> import operator

token_operators = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
binary_ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '^': operator.pow,
}

def parse_expression(tokens, min_precedence=0):
    lhs = parse_primary(tokens)
    
    while tokens and tokens[0] in token_operators and token_operators[tokens[0]] >= min_precedence:
        op = tokens.pop(0)
        precedence = token_operators[op]
        
        rhs = parse_expression(tokens, precedence + 1)
        lhs = binary_ops[op](lhs, rhs)
    
    return lhs

def parse_primary(tokens):
    token = tokens.pop(0)
    if token.isdigit():
        return int(token)
    elif token == '(':
        expr = parse_expression(tokens)
        tokens.pop(0)  # Remove ')'
        return expr
    raise ValueError("Invalid token: " + token)

def tokenize(expression):
    return expression.replace('(', ' ( ').replace(')', ' ) ').split()

def evaluate(expression):
    tokens = tokenize(expression)
    return parse_expression(tokens)

# Example Usage
expr = "3 + 5 * ( 2 - 8 ) ^ 2"
result = evaluate(expr)
print(f"Result: {result}")
____________________________________________________________________________________________________________________________________________________________________________________________

13. Implementation of shift reduce parsing algorithm and LR parser.

=> class LRParser:
    def __init__(self, actions, goto, grammar):
        self.actions, self.goto, self.grammar = actions, goto, grammar

    def parse(self, tokens):
        stack, tokens, idx = [0], tokens + ['$'], 0
        while True:
            state = stack[-1]
            token = tokens[idx] if idx < len(tokens) else '$'
            action = self.actions[state].get(token)

            if not action:
                print("Rejected!")
                return

            if action.startswith("s"): 
                stack.append(int(action[1:])); idx += 1
                print(f"Shift: {stack}")
            elif action.startswith("r"):
                rule_idx = int(action[1:])
                rule_lhs, rule_rhs = self.grammar[rule_idx]
                for _ in rule_rhs: stack.pop()
                new_state = stack[-1]
                goto_state = self.goto[new_state].get(rule_lhs)
                if goto_state is None: 
                    print("Rejected!")
                    return
                stack.append(goto_state)
                print(f"Reduce by {rule_lhs} → {rule_rhs}: {stack}")
            elif action == "acc":
                print("Accepted!")
                return

actions = [
    {'D': 's1'}, {'N': 's2'}, {'V': 's3', '$': 'r0'},
    {'D': 's4'}, {'N': 's5', '$': 'r1'}, {'$': 'acc'}
]
goto = [{'NP': 6}, {}, {}, {'NP': 7}, {}, {}]
grammar = [('S', ['NP', 'VP']), ('NP', ['D', 'N']), ('VP', ['V', 'NP'])]

LRParser(actions, goto, grammar).parse(['D', 'N', 'V', 'D', 'N'])
____________________________________________________________________________________________________________________________________________________________________________________________

14. Write code to generate abstract syntax tree.

=> import ast

code = "x = 5 + 3"
tree = ast.parse(code)
print(ast.dump(tree, indent=4))
___________________________________________________

15	Implement Three Address codes 
def tac(expr):
    p = {'+':1, '-':1, '*':2, '/':2}
    out, op, stack, t = [], [], [], 1

    for x in expr.split():
        if x in p:
            while op and p[op[-1]] >= p[x]: out.append(op.pop())
            op.append(x)
        else:
            out.append(x)
    out += reversed(op)

    for x in out:
        if x in p:
            b, a = stack.pop(), stack.pop()
            print(f"t{t} = {a} {x} {b}")
            stack.append(f"t{t}")
            t += 1
        else: stack.append(x)

tac("a + b * c")
___________________________________________________________________________

16.Implementation of Code Generation

def code_gen(expr):
    ops = {'+':'ADD', '-':'SUB', '*':'MUL', '/':'DIV'}
    stack, t = [], 1
    postfix = expr.split()

    for token in postfix:
        if token in ops:
            b, a = stack.pop(), stack.pop()
            print(f"T{t} = {a}")
            print(f"{ops[token]} T{t}, {b}")
            stack.append(f"T{t}")
            t += 1
        else:
            stack.append(token)

# Example with postfix: "a b + c *"
code_gen("a b + c *")

