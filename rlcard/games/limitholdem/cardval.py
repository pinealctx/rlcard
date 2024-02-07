from enum import Enum


class RankCategory(Enum):
    STRAIGHT_FLUSH = 8
    FOUR_KIND = 7
    FULL_HOUSE = 6
    FLUSH = 5
    STRAIGHT = 4
    THREE_KIND = 3
    TWO_PAIR = 2
    ONE_PAIR = 1
    HIGH_CARD = 0


class CardValue(object):
    """
       牌面分数编码
       6*4 bits
       20-23 bits      |     16-19 bits |     12-15bits |     8-11 bits |        4-7 bits |    0-3 bits
       category rank   |
       straight flush  |             NA |            NA |            NA |              NA |    straight rank(start card)
                                                                                                 A2345 : 1 ... TJQKA : 10
       four kind       |             NA |            NA |            NA |              NA |    single card in four kind
       full house      |             NA |            NA |            NA |     three cards |    two cards
       flush           |      1st card  |      2nd card | 3rd      card |        4th card |    5th card
       straight        |             NA |            NA |            NA |              NA |    straight rank(start card)
                                                                                                 A2345 : 1 ... TJQKA : 10
       three kind      |             NA |            NA |   three cards | 1st single card |    2nd single card
       two pair        |             NA |            NA | 1st two cards |  2nd two cards  |    single card
       one pair        |             NA |     two cards |      1st card |       2nd card  |    3th card
       high card       |      1st card  |      2nd card |      3rd card |        4th card |    5th card
    """

    def __init__(self, rank):
        self.rank = rank << 20

    def __init__(self, rank, *cards):
        """cards, 小牌先入栈"""
        self.rank = rank << 20
        for i, card in enumerate(cards):
            self.rank += card << (4 * i)

    def add_card(self, card, idx=0):
        self.rank += card << (4 * idx)
        return self

    def add_cards(self, cards: list):
        """from low to high"""
        for i, card in cards:
            self.rank += card << (4 * i)
        return self

    def add_card_v2(self, *cards):
        """cards, 小牌先入栈"""
        for i, card in enumerate(cards):
            self.rank += card << (4 * i)
        return self
