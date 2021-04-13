"""
Created on 26 oct. 16:01 2020

@author: HaroldKS
"""

from core import Player
from seega.seega_rules import SeegaRules

class AI(Player):

    in_hand = 12
    score = 0

    def __init__(self, color):
        super(AI, self).__init__(color)
        self.position = color.value

    def play(self, state, remain_time):
        # print(f"Player {self.position} is playing.")
        # print("time remain is ", remain_time, " seconds")
        return SeegaRules.random_play(state, self.position)

    def set_score(self, new_score):
        self.score = new_score

    def update_player_infos(self, infos):
        self.in_hand = infos['in_hand']
        self.score = infos['score']
        
    def reset_player_informations(self):
        self.in_hand = 12
        self.score = 0
