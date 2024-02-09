import numpy as np

from collections import Counter
from enum import Enum

"""card category"""
card_category = {
    1: 'High Card', 2: 'One Pair', 3: 'Two Pair', 4: 'Three of a Kind',
    5: 'Straight', 6: 'Flush', 7: 'Full House', 8: 'Four of a Kind',
    9: 'Straight Flush',
}

"""card number index code"""
card_idx_code = {
    'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12
}

"""card rank code"""
card_rank_code = [14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

"""straight idx code"""
"""use mask 13 length list to represent straight idx"""
straight_idx_code_mask = [
    # TJQKA
    # A  2  3  4  5  6  7  8  9  T  J  Q  K
    ([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], 10),
    # 9TJQK
    # A  2  3  4  5  6  7  8  9  T  J  Q  K
    ([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 9),
    # 89TJQ
    # A  2  3  4  5  6  7  8  9  T  J  Q  K
    ([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], 8),
    # 789TJ
    # A  2  3  4  5  6  7  8  9  T  J  Q  K
    ([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], 7),
    # 6789T
    # A  2  3  4  5  6  7  8  9  T  J  Q  K
    ([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], 6),
    # 56789
    # A  2  3  4  5  6  7  8  9  T  J  Q  K
    ([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], 5),
    # 45678
    # A  2  3  4  5  6  7  8  9  T  J  Q  K
    ([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 4),
    # 34567
    # A  2  3  4  5  6  7  8  9  T  J  Q  K
    ([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 3),
    # 23456
    # A  2  3  4  5  6  7  8  9  T  J  Q  K
    ([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 2),
    # A2345
    # A  2  3  4  5  6  7  8  9  T  J  Q  K
    ([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 1),
]


class Color(Enum):
    CLUB = 1
    DIAMOND = 2
    HEART = 3
    SPADE = 4

    @classmethod
    def from_string(cls, color: str):
        if color == 'C':
            return cls.CLUB
        elif color == 'D':
            return cls.DIAMOND
        elif color == 'H':
            return cls.HEART
        else:
            return cls.SPADE

    def __str__(self):
        if self == Color.CLUB:
            return 'C'
        elif self == Color.DIAMOND:
            return 'D'
        elif self == Color.HEART:
            return 'H'
        else:
            return 'S'

    def __repr__(self):
        return self.__str__()


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

    def __init__(self, rank: RankCategory):
        self.rank = rank << 20

    def __init__(self, rank: RankCategory, *cards):
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

    def add_np_cards(self, cards: np.ndarray):
        """from low to high"""
        for i, card in enumerate(cards):
            self.rank += card << (4 * i)
        return self

    def add_card_v2(self, *cards):
        """cards, 小牌先入栈"""
        for i, card in enumerate(cards):
            self.rank += card << (4 * i)
        return self


class CardComb(object):
    """
    use 4*14 matrix to represent the card combination
    Actually, considering the straight, we use 14 to represent card number.(A1234...JQKA).
    'A' is loop here.
    0: diamond
    1: club
    2: heart
    3: spade
    """

    def __init__(self):
        self.card_matrix = np.zeros((4, 14), dtype=np.int8)
        # 行列统计
        self.row_counter = np.zeros(4, dtype=np.int8)
        self.col_counter = np.zeros(14, dtype=np.int8)
        # 同花花色index
        self.flush_idx = -1
        # 列统计计数
        self.card4_idx = np.empty(1, dtype=np.int8)
        self.card3_idx = np.empty(3, dtype=np.int8)
        self.card2_idx = np.empty(3, dtype=np.int8)
        self.card1_idx = np.empty(7, dtype=np.int8)

    def add_card(self, card: str):
        """use numpy method"""
        card_color = Color.from_string(card[0])
        card_idx = card_idx_code[card[1]]

        if card_idx == 0:
            # if card is 'A', add it to the end of the matrix
            self.card_matrix[card_color.value - 1, -1] = 1
        self.card_matrix[card_color.value - 1, card_idx] = 1
        return self

    def add_cards(self, cards: list):
        """use numpy method"""
        for card in cards:
            self.add_card(card)
        return self

    def do_count(self):
        """计算牌力"""
        # 先统计列
        self.count_col()

        # 同花顺/同花/顺子
        if self.can_count_straight_or_flush():
            # 统计行(剔除第一列)
            self.count_row()

            # 有同花的情况
            if self.flush_idx != -1:
                # 优先统计同花顺
                value = self.count_straight_flush(self.flush_idx)
                if value != -1:
                    return value
                # 没有同花顺，统计同花
                # 从最后的idx开始到1，找到5个大于等于1的idx就可以组成同花高牌
                return self.count_high_card(CardValue(RankCategory.FLUSH), self.card_matrix[self.flush_idx, :])
            # 统计顺子
            value = self.count_straight()
            if value != -1:
                return value

        # 按优先级来 4条/葫芦/3条/2对/1对/高牌
        return self.count_misc_combine()

    def count_col(self):
        """按列统计"""
        self.col_counter = np.sum(self.card_matrix, axis=0)
        for i in range(1, 14):
            if self.col_counter[i] == 4:
                self.card4_idx = np.append(self.card4_idx, i)
            elif self.col_counter[i] == 3:
                self.card3_idx = np.append(self.card3_idx, i)
            elif self.col_counter[i] == 2:
                self.card2_idx = np.append(self.card2_idx, i)
            elif self.col_counter[i] == 1:
                self.card1_idx = np.append(self.card1_idx, i)

    def count_row(self):
        """按行统计,不包括第一列"""
        self.row_counter = np.sum(self.card_matrix[:, 1:], axis=1)
        s_color = np.where(self.row_counter >= 5)
        if len(s_color) > 0:
            # 有同花
            self.flush_idx = s_color[0][0]

    def can_count_straight_or_flush(self):
        """判断是否需要进行顺子或同花的判断"""
        # 1. 有4张时肯定没有顺子或同花
        # 2. 有2个3张时肯定没有顺子或同花
        # 3. 有1个3张和1个2张时肯定没有顺子或同花
        # 4. 有3个2张时肯定没有顺子或同花
        # 5. 换个角度，单张牌计数大于等于3时才可能有顺子或同花
        return len(self.card1_idx) >= 3

    def count_straight_flush(self, flush_idx: int):
        """判断同花顺"""
        for i in range(10):
            if np.all(self.card_matrix[flush_idx, i: i + 5] == 1):
                return CardValue(RankCategory.STRAIGHT_FLUSH).add_card(i + 1).rank
        return -1

    def count_straight(self):
        """判断顺子"""
        for i in range(10):
            if np.all(self.col_counter[i: i + 5] >= 1):
                return CardValue(RankCategory.STRAIGHT).add_card(i + 1).rank
        return -1

    def count_misc_combine(self):
        """依次优先级统计4条/葫芦/3条/2对/1对/高牌"""
        card3_size = len(self.card3_idx)
        card2_size = len(self.card2_idx)
        if len(self.card4_idx) > 0:
            return self.count_four_kind(CardValue(RankCategory.FOUR_KIND), self.card4_idx[0], self.col_counter)
        elif card3_size == 2:
            # 2个3条，肯定是葫芦
            return CardValue(RankCategory.FULL_HOUSE).add_np_cards(self.card3_idx).rank
        elif card3_size == 1:
            if card2_size >= 1:
                # 1个3条和1个或2个2条，肯定是葫芦
                return CardValue(RankCategory.FULL_HOUSE).add_card_v2(
                    self.card2_idx[-1],
                    self.card3_idx[0],
                ).rank
            else:
                # 1个3条，剩下的是单牌
                return CardValue(RankCategory.THREE_KIND).add_card_v2(
                    self.card1_idx[-2],
                    self.card1_idx[-1],
                    self.card3_idx[0],
                ).rank
        elif card2_size == 2:
            # 2个2条
            return CardValue(RankCategory.TWO_PAIR).add_card_v2(
                self.card1_idx[-1],
                self.card2_idx[0],
                self.card2_idx[1],
            ).rank
        elif card2_size == 1:
            # 1个2条
            return CardValue(RankCategory.ONE_PAIR).add_card_v2(
                self.card1_idx[-3],
                self.card1_idx[-2],
                self.card1_idx[-1],
                self.card2_idx[0],
            ).rank
        else:
            # 高牌
            return self.count_high_card(CardValue(RankCategory.HIGH_CARD), self.col_counter)

    @staticmethod
    def count_high_card(rank_cat: CardValue, arr: np.ndarray):
        """统计高牌"""
        # 从最后的idx开始，找到5个等于1的idx就可以组成高牌
        return rank_cat.add_cards(np.argwhere(arr == 1).flatten()[-5:]).rank

    @staticmethod
    def count_four_kind(rank_cat: CardValue, idx: int, arr: np.ndarray):
        """统计4张-找到4张组合最大的单牌"""
        # 倒着查找
        for i in range(13, -1, 0):
            if 4 > arr[i] > 0:
                return rank_cat.add_card_v2(i, idx + 1).rank
        raise ValueError('count_four_kind error')
