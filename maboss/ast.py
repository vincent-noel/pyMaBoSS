"""utilitary functions to manipulate boolean expressions."""


from sys import stderr
import pyparsing as pp
import json

class ASTNode(object):

    def __init__(self, tokens=None):

        if tokens is not None:
            self.tokens = tokens
            self.identify()

    def identify(self):

        if 'tokens' in self.__dict__.keys():
            while len(self.tokens) == 1 and not (isinstance(self.tokens[0], str) or isinstance(self.tokens[0], float)) and 'tokens' in self.tokens[0].__dict__.keys():
                self.tokens = self.tokens[0].tokens

            if len(self.tokens) == 1 and isinstance(self.tokens[0], float) or isinstance(self.tokens[0], int):
                self.__class__ = ASTReal

            elif len(self.tokens) == 1 and isinstance(self.tokens[0], str):
                if self.tokens[0].lower() in ['true', 'false']:
                    self.__class__ = ASTBool
                else:
                    self.__class__ = ASTVar

            elif self.tokens[0] in ['~', '!', 'not', 'NOT']:
                self.__class__ = ASTNot

            elif len(self.tokens) > 1 and self.tokens[1] in ['&&', '&', 'AND']:
                self.__class__ = ASTAnd

            elif len(self.tokens) > 1 and self.tokens[1] in ['||', '|', 'OR']:
                self.__class__ = ASTOr

            elif len(self.tokens) > 1 and self.tokens[1] in ['^', 'XOR']:
                self.__class__ = ASTXor

            elif len(self.tokens) > 1 and self.tokens[1] == '?':
                self.__class__ = ASTIFE

            elif len(self.tokens) > 1 and self.tokens[0] == "(" and self.tokens[2] == ")":

                if isinstance(self.tokens[1], str):
                    t_node = ASTNode()
                    t_node.loadFromString(self.tokens[1].replace('\'', '"'))
                    self.__class__ = self.tokens.asDict()['AST'].__class__
                    self.__dict__ = self.tokens.asDict()['AST'].__dict__

                else:
                    self.__class__ = self.tokens[1].__class__
                    self.__dict__ = self.tokens[1].__dict__

                return

            self.assignFields()

    def loadFromDict(self, read_dict):

        self.operator = read_dict['operator']
        self.args = []
        for arg in read_dict['args']:
            if not (isinstance(arg, str) or isinstance(arg, float) or isinstance(arg, int)):
                new_node = ASTNode()
                new_node.loadFromDict(arg)
                self.args.append(new_node)
            else:
                self.args.append(arg)

        if self.operator == 'not':
            self.__class__ = ASTNot
        elif self.operator == 'and':
            self.__class__ = ASTAnd
        elif self.operator == 'or':
            self.__class__ = ASTOr
        elif self.operator == 'xor':
            self.__class__ = ASTXor
        elif self.operator == 'ife':
            self.__class__ = ASTIFE
        elif self.operator == 'var':
            self.__class__ = ASTVar
        elif self.operator == 'bool':
            self.__class__ = ASTBool
        elif self.operator == 'real':
            self.__class__ = ASTReal
        else:
            print("Unkown ASTNode type when importing from dict : %s" % self.operator)


    def loadFromString(self, read_string):

        t_dict = json.loads(read_string)
        self.loadFromDict(t_dict)

    def __str__(self):
        return str(self.__dict__)

    __repr__ = __str__

    def assignFields(self):
        self.__class__ = self.tokens[0].__class__
        self.__dict__ = self.tokens[0].__dict__

    def listVars(self):
        vars = []
        if self.operator == 'var':
            vars.append(self.args[0])
        elif not (self.operator == 'real' or self.operator == 'bool'):
            for arg in self.args:
                vars += arg.listVars()
        return vars

    def copy(self):
        new_node = ASTNode()
        new_node.__class__ = self.__class__
        new_node.operator = self.operator
        new_node.args = []
        for arg in self.args:
            if isinstance(arg, ASTNode):
                new_node.args.append(arg.copy())
            else:
                new_node.args.append(arg)
        return new_node

    def subs(self, logic, internal_vars):

        if self.operator == 'var':

            if self.args[0] == '@logic':
                t_ast = logic.copy()
                self.__class__ = t_ast.__class__
                self.__dict__ = t_ast.__dict__

            elif self.args[0] in internal_vars.keys():
                t_ast = internal_vars[self.args[0]].copy()
                self.__class__ = t_ast.__class__
                self.__dict__ = t_ast.__dict__

        elif self.operator not in ['real', 'bool']:
            for arg in self.args:
                arg.subs(logic, internal_vars)

    def develop(self, logic, internal_vars):
        new_node = self.copy()
        new_node.subs(logic, internal_vars)
        return new_node



class ASTReal(ASTNode):

    def assignFields(self):
        self.operator = 'real'
        self.args = [self.tokens[0]]
        del self.tokens

    def __str__(self):
        return str(self.args[0])

class ASTBool(ASTNode):

    def assignFields(self):
        self.operator = 'bool'
        self.args = [self.tokens[0].upper()]
        del self.tokens

    def __str__(self):
        return str(self.args[0])


class ASTVar(ASTNode):

    def assignFields(self):
        self.operator = 'var'
        self.args = [self.tokens[0]]
        del self.tokens

    def __str__(self):
        return self.args[0]


class ASTNot(ASTNode):

    def assignFields(self):
        self.operator = 'not'
        self.args = [self.tokens[1]]
        del self.tokens


class ASTAnd(ASTNode):

    def assignFields(self):
        self.operator = 'and'
        self.args = self.tokens[::2]
        del self.tokens


class ASTOr(ASTNode):

    def assignFields(self):
        self.args = self.tokens[::2]
        self.operator = 'or'
        del self.tokens


class ASTXor(ASTNode):

    def assignFields(self):
        self.args = self.tokens[::2]
        self.operator = 'xor'
        del self.tokens


class ASTIFE(ASTNode):

    def assignFields(self):
        self.operator = 'ife'
        self.args = self.tokens[::2]
        del self.tokens


logExp = pp.Forward()

boolCst = pp.oneOf("True False")
realCst = pp.pyparsing_common.real.copy()
boolNot = pp.oneOf("! ~ NOT")
boolAnd = pp.oneOf("&& & AND")
boolOr = pp.oneOf("|| | OR")
boolXor = pp.oneOf("^ XOR")
boolTest = pp.Literal("?")
boolElse = pp.Literal(":")
lparen = '('
rparen = ')'

varName = (
        ~boolAnd + ~boolOr + ~boolXor + ~boolNot + ~boolCst + ~realCst + ~boolTest
        + ~boolElse + ~pp.Literal('Node') + pp.Word(pp.alphas+'$'+'@', pp.alphanums+'_')
).setParseAction(ASTNode)
logTerm = (boolCst | realCst | varName | (lparen + logExp + rparen)).setParseAction(ASTNode)
logNot = (pp.ZeroOrMore(boolNot) + logTerm).setParseAction(ASTNode)
logAnd = (logNot + pp.ZeroOrMore(boolAnd + logNot)).setParseAction(ASTNode)
logOr = (logAnd + pp.ZeroOrMore(boolOr + logAnd)).setParseAction(ASTNode)
logXor = (logOr + pp.ZeroOrMore(boolXor + logOr)).setParseAction(ASTNode)
logIFE = (logXor + pp.ZeroOrMore(boolTest + logXor + boolElse + logXor)).setParseAction(ASTNode).setResultsName("AST")
logExp << pp.Combine(logIFE, adjacent=False, joinString=' ')

