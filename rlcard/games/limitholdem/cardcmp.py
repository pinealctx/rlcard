import random
import time

import numpy as np

from enum import Enum

from rlcard.games.limitholdem.utils import compare_hands, Hand


"""card number index code"""
card_idx_code = {
    'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12
}

"""card code idx"""
card_code_idx = {
    0: 'A', 13: 'A', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: 'T', 10: 'J', 11: 'Q', 12: 'K'
}

"""rank levels"""
rank_levels = [
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
]

"""all cards"""""
whole_cards = ['CA', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'CT', 'CJ', 'CQ', 'CK',
               'DA', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'DT', 'DJ', 'DQ', 'DK',
               'HA', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'HT', 'HJ', 'HQ', 'HK',
               'SA', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'ST', 'SJ', 'SQ', 'SK',
               ]

cards_idx = {
    'CA': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7, 'C8': 8, 'C9': 9, 'CT': 10, 'CJ': 11, 'CQ': 12, 'CK': 13,
    'DA': 14, 'D2': 15, 'D3': 16, 'D4': 17, 'D5': 18, 'D6': 19, 'D7': 20, 'D8': 21, 'D9': 22, 'DT': 23, 'DJ': 24, 'DQ': 25, 'DK': 26,
    'HA': 27, 'H2': 28, 'H3': 29, 'H4': 30, 'H5': 31, 'H6': 32, 'H7': 33, 'H8': 34, 'H9': 35, 'HT': 36, 'HJ': 37, 'HQ': 38, 'HK': 39,
    'SA': 40, 'S2': 41, 'S3': 42, 'S4': 43, 'S5': 44, 'S6': 45, 'S7': 46, 'S8': 47, 'S9': 48, 'ST': 49, 'SJ': 50, 'SQ': 51, 'SK': 52,
}


def seven_cards_encode(cards):
    """7张牌编码"""
    cs = [cards_idx[card] for card in cards]
    # sort cs
    cs.sort()
    # 编码成一个整数
    return cs[0] + (cs[1] << 8) + (cs[2] << 16) + (cs[3] << 24) + (cs[4] << 32) + (cs[5] << 40) + (cs[6] << 48)


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


class PreAllocNpArray(object):
    def __init__(self, size):
        self.size = size
        self.arr = np.empty(size, dtype=np.int8)
        self.idx = 0

    def add(self, value):
        self.arr[self.idx] = value
        self.idx += 1
        return self

    def reset(self):
        self.idx = 0
        return self

    def get(self):
        return self.arr[:self.idx]

    def __len__(self):
        return self.idx

    def __getitem__(self, item):
        """support negative index"""
        if item < 0:
            return self.arr[self.idx + item]
        else:
            return self.arr[item]


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

    def __str__(self):
        return rank_levels[self.value]


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
       one pair        |             NA |     two cards |      1st card |       2nd card  |    3rd card
       high card       |      1st card  |      2nd card |      3rd card |        4th card |    5th card
    """

    def __init__(self, rank: RankCategory):
        self.rank = rank.value << 20

    def add_card(self, card, idx=0):
        self.rank += card << (4 * idx)
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

    @staticmethod
    def value_to_str(value):
        cat_rank = value >> 20
        cat_rank_str = rank_levels[cat_rank]
        for i in range(5):
            card = (value >> (4 * i)) & 0xF
            if card != 0:
                cat_rank_str += f' {card_code_idx[card]}'
        return cat_rank_str


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
        self.card4_idx = PreAllocNpArray(1)
        self.card3_idx = PreAllocNpArray(2)
        self.card2_idx = PreAllocNpArray(3)
        self.card1_idx = PreAllocNpArray(7)

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

    def count_value(self):
        """计算牌力"""
        # 先统计列
        self.count_col()

        # 同花顺/同花/顺子
        value = self.count_flush_or_straight_or_both()
        if value != -1:
            return value
        # 按优先级来 4条/葫芦/3条/2对/1对/高牌
        return self.count_misc_combine()

    def count_col(self):
        """按列统计"""
        self.col_counter = np.sum(self.card_matrix, axis=0)
        for i in range(1, 14):
            if self.col_counter[i] == 4:
                self.card4_idx.add(i)
            elif self.col_counter[i] == 3:
                self.card3_idx.add(i)
            elif self.col_counter[i] == 2:
                self.card2_idx.add(i)
            elif self.col_counter[i] == 1:
                self.card1_idx.add(i)

    def count_row(self):
        """按行统计,不包括第一列"""
        self.row_counter = np.sum(self.card_matrix[:, 1:], axis=1)
        s_color = np.where(self.row_counter >= 5)
        if s_color[0].size > 0:
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

    def count_flush_or_straight_or_both(self):
        """统计同花顺/同花/顺子"""
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
            return self.count_straight()
        return -1

    def count_straight_flush(self, flush_idx: int):
        """判断同花顺"""
        for i in range(9, -1, -1):
            if np.all(self.card_matrix[flush_idx, i: i + 5] == 1):
                return CardValue(RankCategory.STRAIGHT_FLUSH).add_card(i).rank
        return -1

    def count_straight(self):
        """判断顺子"""
        for i in range(9, -1, -1):
            if np.all(self.col_counter[i: i + 5] >= 1):
                return CardValue(RankCategory.STRAIGHT).add_card(i).rank
        return -1

    def count_misc_combine(self):
        """依次优先级统计4条/葫芦/3条/2对/1对/高牌"""
        card3_size = len(self.card3_idx)
        card2_size = len(self.card2_idx)
        if len(self.card4_idx) > 0:
            return self.count_four_kind(CardValue(RankCategory.FOUR_KIND), self.card4_idx[0], self.col_counter)
        elif card3_size == 2:
            # 2个3条，肯定是葫芦
            return CardValue(RankCategory.FULL_HOUSE).add_np_cards(self.card3_idx.get()).rank
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
        elif card2_size == 3:
            # 3个2条
            max_card = max(self.card2_idx[0], self.card1_idx[0])
            return CardValue(RankCategory.TWO_PAIR).add_card_v2(
                max_card,
                self.card2_idx[1],
                self.card2_idx[2],
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
        return rank_cat.add_np_cards(np.argwhere(arr == 1).flatten()[-5:]).rank

    @staticmethod
    def count_four_kind(rank_cat: CardValue, idx: int, arr: np.ndarray):
        """统计4张-找到4张组合最大的单牌"""
        # 倒着查找
        for i in range(13, -1, -1):
            if 4 > arr[i] > 0:
                return rank_cat.add_card_v2(i, idx).rank
        raise ValueError('count_four_kind error')


def compare_hands_v1(player1, player2, community_cards):
    """比较两名玩家的手牌"""
    player1_cards = player1 + community_cards
    player2_cards = player2 + community_cards
    winner = compare_hands([player1_cards, player2_cards])
    if winner[0] == 1:
        if winner[1] == 1:
            return 0
        else:
            return 1
    else:
        return -1


def debug_compare_hands_v1(p1, p2, community_cards):
    h1, h2 = Hand(p1 + community_cards), Hand(p2 + community_cards)
    h1.evaluateHand()
    h2.evaluateHand()
    print("player1:", h1.best_five)
    print("player2:", h2.best_five)


def compare_hands_v2(player1, player2, community_cards):
    player1_comb = CardComb().add_cards(player1 + community_cards)
    player2_comb = CardComb().add_cards(player2 + community_cards)

    player1_value = player1_comb.count_value()
    player2_value = player2_comb.count_value()

    if player1_value > player2_value:
        return 1
    elif player1_value < player2_value:
        return -1
    else:
        return 0


def debug_compare_hands_v2(p1, p2, community_cards):
    player1_comb = CardComb().add_cards(p1 + community_cards)
    player2_comb = CardComb().add_cards(p2 + community_cards)
    player1_value = player1_comb.count_value()
    player2_value = player2_comb.count_value()

    print("player1:", CardValue.value_to_str(player1_value), player1_value)
    print("player2:", CardValue.value_to_str(player2_value), player2_value)


def compare_v1_v2(num_trials):
    for i in range(num_trials):
        random_cards = random.sample(whole_cards, 9)
        player1_hand, player2_hand, community_cards = \
            random_cards[:2], random_cards[2:4], random_cards[4:]
        cp1 = compare_hands_v1(player1_hand, player2_hand, community_cards)
        cp2 = compare_hands_v2(player1_hand, player2_hand, community_cards)
        if cp1 != cp2:
            print(player1_hand, player2_hand, community_cards, cp1, cp2)
            debug_compare_hands_v1(player1_hand, player2_hand, community_cards)
            debug_compare_hands_v2(player1_hand, player2_hand, community_cards)
        if i % 1000000 == 0:
            print('num_trials:', i)


def compare_v1_v2_time(fn, num_trials):
    start = time.time()
    for i in range(num_trials):
        random_cards = random.sample(whole_cards, 9)
        player1_hand, player2_hand, community_cards = \
            random_cards[:2], random_cards[2:4], random_cards[4:]
        fn(player1_hand, player2_hand, community_cards)
    end = time.time()
    print(fn.__name__, end - start, "avg:", (end - start) / num_trials)


def count_player_hand(hand):
    comb = CardComb().add_cards(hand)
    value = comb.count_value()
    print("value:", value, CardValue.value_to_str(value))


if __name__ == '__main__':
    # debug_compare_hands_v2(['D9', 'D2'], ['S4', 'S9'], ['S3', 'D4', 'C9', 'DQ', 'D8'])
    # debug_compare_hands_v2(['DQ', 'D7'], ['C6', 'D9'], ['SA', 'H6', 'H9', 'CK', 'HA'])
    # count_player_hand(['C4', 'CK', 'CA', 'SK', 'DT', 'D4', 'SA'])
    # compare_v1_v2_time(compare_hands_v1, 100000)
    # compare_v1_v2_time(compare_hands_v2, 100000)
    compare_v1_v2(1000000 * 100)
