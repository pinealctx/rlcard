import numpy as np

from collections import Counter
from enum import Enum

from .cardval import RankCategory, CardValue

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
        value = 0

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
        card3_size = len(self.card3_idx)
        card2_size = len(self.card2_idx)
        if len(self.card4_idx) > 0:
            return self.count_four_kind(CardValue(RankCategory.FOUR_KIND), self.card4_idx[0], self.col_counter)
        elif card3_size == 2:
            # 2个3条，肯定是葫芦
            card3s = self.card3_idx.tolist()
            return CardValue(RankCategory.FULL_HOUSE).add_cards(card3s).rank
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

        # 统计顺子
        for i in range(10):
            if np.all(self.col_counter[i: i + 5] == 1):
                self.straight[i] = 1
                self.has_straight = True
        # 先看同花顺
        if self.has_straight and self.has_flush:
            for i in range(4):
                if np.all(self.card_matrix[i, :] == 1):
                    return CardValue(RankCategory.STRAIGHT_FLUSH, 14).rank
        return self

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

    def count_straight_flush(self, flush_idx):
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

    @staticmethod
    def count_high_card(rank_cat, arr):
        """统计高牌"""
        # 从最后的idx开始，找到5个等于1的idx就可以组成高牌
        return rank_cat.add_cards(np.argwhere(arr == 1).flatten()[-5:]).rank

    @staticmethod
    def count_four_kind(rank_cat, idx, arr):
        """统计4张-找到4张组合最大的单牌"""
        # 倒着查找
        for i in range(13, -1, 0):
            if 4 > arr[i] > 0:
                return rank_cat.add_card_v2(i, idx + 1).rank
        raise ValueError('count_four_kind error')


'''
    def straight_flush_rank(self):
        """in same color and continuous"""
        for i in range(4):
            for mask in straight_idx_code_mask:
                if np.all(self.card_matrix[i, :] == mask):
                    return CardValue(RankCategory.STRAIGHT_FLUSH, mask[1]).rank
        return -1

    def four_kind_rank(self):
        """four same number"""
        for i in range(13):
            if np.sum(self.card_matrix[:, i]) == 4:
                # find four some number and the left one

        return -1

    def full_house_rank(self):
        """three same number and one pair"""
        three_kind = -1
        one_pair = -1
        for i in range(13):
            s = np.sum(self.card_bytes[i: 52: 13])
            if s == 3:
                three_kind = i+1
            elif s == 2:
                one_pair = i+1
        if three_kind != -1 and one_pair != -1:
            return three_kind
        return -1
'''


class FiveCard(object):
    def __init__(self):
        self.cards = []
        self.suits = []
        self.ranks = []
        self.card_count = {}
        self.suit_count = {}
        self.rank_count = {}
        self.straight = False
        self.flush = False
        self.straight_flush = False
        self.royal_flush = False
        self.four_kind = False
        self.three_kind = False
        self.two_pair = False
        self.one_pair = False
        self.high_card = False
        self.rank = 0
        self.hand = None
        self.hand_name = None
        self.hand_rank = None
        self.hand_rank_name = None

    def get_hand(self, cards):
        self.cards = cards
        self.suits = [card[1] for card in cards]
        self.ranks = [card[0] for card in cards]
        self.card_count = Counter(self.cards)
        self.suit_count = Counter(self.suits)
        self.rank_count = Counter(self.ranks)
        self.is_straight()
        self.is_flush()
        self.is_straight_flush()
        self.is_royal_flush()
        self.is_four_kind()
        self.is_three_kind()
        self.is_two_pair()
        self.is_one_pair()
        self.is_high_card()
        self.get_hand_rank()
        self.get_hand_name()
        return self.hand

    def is_straight(self):
        if len(self.rank_count) == 5:
            if max(self.ranks) - min(self.ranks) == 4:
                self.straight = True
                return True
        return False

    def is_flush(self):
        if len(self.suit_count) == 1:
            self.flush = True
            return True
        return False

    def is_straight_flush(self):
        if self.straight and self.flush:
            self.straight_flush = True
            return True
        return False

    def is_royal_flush(self):
        if self.straight_flush and max(self.ranks) == 14:
            self.royal_flush = True
            return True
        return False

    def is_four_kind(self):
        if 4 in self.rank_count.values():
            self.four_kind = True
            return True
        return False

    def is_three_kind(self):
        if 3 in self.rank_count.values():
            self.three_kind = True
            return True
        return False

    def is_two_pair(self):
        if list(self.rank_count.values()).count(2) == 2:
            self.two_pair = True
            return True
        return False

    def is_one_pair(self):
        if 2 in self.rank_count.values():
            self.one_pair = True
            return True
        return False

    def is_high_card(self):
        self.high_card = True
        return True

    def get_hand_rank(self):
        if self.royal_flush:
            self.hand_rank = 10
        elif self.straight_flush:
            self.hand_rank = 9
        elif self.four_kind:
            self.hand_rank = 8
        elif self.three_kind and self.one_pair:
            self.hand_rank = 7
        elif self.flush:
            self.hand_rank = 6
        elif self.straight:
            self.hand_rank = 5
        elif self.three_kind:
            self.hand_rank = 4
        elif self.two_pair:
            self.hand_rank = 3
        elif self.one_pair:
            self.hand_rank = 2
        else:
            self.hand_rank = 1
        return self.hand_rank

    def get_hand_name(self):
        if self.royal_flush:
            self.hand_name = 'Royal Flush'
        elif self.straight_flush:
            self.hand_name = 'Straight Flush'
        elif self.four_kind:
            self.hand_name = 'Four of a Kind'
        elif self.three_kind and self.one_pair:
            self.hand_name = 'Full House'
        elif self.flush:
            self.hand_name = 'Flush'
        elif self.straight:
            self.hand_name = 'Straight'
        elif self.three_kind:
            self.hand_name = 'Three of a Kind'
        elif self.two_pair:
            self.hand_name = 'Two Pair'
        elif self.one_pair:
            self.hand_name = 'One Pair'
        else:
            self.hand_name = 'High Card'
        return self.hand_name

    def get_hand(self, cards):
        self.hand = (self.hand_rank, self.hand_name)
        return self.hand

    def compare(self, hand1, hand2):
        if hand1[0] > hand2[0]:
            return 1
        elif hand1[0] < hand2[0]:
            return -1
        else:
            return 0
