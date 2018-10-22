"""utilitary functions to manipulate boolean expressions."""


from sys import stderr
import pyparsing as pp
import json

class ASTNode(object):

    def __init__(self, tokens):
        self.tokens = tokens
        self.identify()

    def identify(self):

        if 'tokens' in self.__dict__.keys():
            while len(self.tokens) == 1 and not isinstance(self.tokens[0], str) and 'tokens' in self.tokens[0].__dict__.keys():
                self.tokens = self.tokens[0].tokens

            if len(self.tokens) == 1 and isinstance(self.tokens[0], str):
                self.__class__ = ASTVar

            elif self.tokens[0] in ['~', '!', 'not', 'NOT']:
                self.__class__ = ASTNot

            elif len(self.tokens) > 1 and self.tokens[1] in ['&&', '&']:
                self.__class__ = ASTAnd

            elif len(self.tokens) > 1 and self.tokens[1] in ['||', '|']:
                self.__class__ = ASTOr

            elif len(self.tokens) > 1 and self.tokens[1] in ['^', 'XOR']:
                self.__class__ = ASTXor

            elif len(self.tokens) > 1 and self.tokens[1] == '?':
                self.__class__ = ASTIFE

            elif len(self.tokens) > 1 and self.tokens[0] == "(" and self.tokens[2] == ")":

                self.__dict__ = json.loads(self.tokens[1].replace('\'', '"'))
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

                return

            self.assignFields()

    def __str__(self):
        return str(self.__dict__)

    __repr__ = __str__

    def assignFields(self):
        self.__class__ = self.tokens[0].__class__
        self.__dict__ = self.tokens[0].__dict__


class ASTVar(ASTNode):

    def assignFields(self):
        self.operator = 'var'
        self.args = self.tokens[0]
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
boolNot = pp.oneOf("! ~ NOT")
boolAnd = pp.oneOf("&& & AND")
boolOr = pp.oneOf("|| | OR")
boolXor = pp.oneOf("^ XOR")
boolTest = pp.Literal("?")
boolElse = pp.Literal(":")
lparen = '('
rparen = ')'

varName = (
        ~boolAnd + ~boolOr + ~boolXor + ~boolNot + ~boolCst + ~boolTest
        + ~boolElse + ~pp.Literal('Node') + pp.Word(pp.alphas+'$', pp.alphanums+'_')
).setParseAction(ASTNode)
logTerm = (boolCst | varName | (lparen + logExp + rparen).setParseAction(ASTNode)).setParseAction(ASTNode)
logNot = (pp.ZeroOrMore(boolNot) + logTerm).setParseAction(ASTNode)
logAnd = (logNot + pp.ZeroOrMore(boolAnd + logNot)).setParseAction(ASTNode)
logOr = (logAnd + pp.ZeroOrMore(boolOr + logAnd)).setParseAction(ASTNode)
logXor = (logOr + pp.ZeroOrMore(boolXor + logOr)).setParseAction(ASTNode)
logIFE = (logXor + pp.ZeroOrMore(boolTest + logXor + boolElse + logXor)).setParseAction(ASTNode).setResultsName("AST")
logExp << pp.Combine(logIFE, adjacent=False, joinString=' ')

