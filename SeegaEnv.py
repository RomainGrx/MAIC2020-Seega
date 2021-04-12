#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 12, 13:48:50
@last modified : 2021 Apr 12, 19:51:19
"""

from seega import SeegaState
from core import Player, Color, Board
from seega import SeegaRules
from seega import SeegaAction
from seega.seega_actions import SeegaActionType
from copy import deepcopy
from utils.timer import Timer
from utils.trace import Trace
import argparse
import sys

import operator
from enum import Enum

import gym
from gym import Env, spaces

import numpy as np


class Action(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    @staticmethod
    def get_dx(action):
        return [(-1, 0), (0, 1), (1, 0), (0, -1)][action]


class Reward(Enum):
    ILLEGAL_MOVE = -10
    PLAYER_PART = 1
    OPONENT_PART = -0.5


class SeegaEnv(Env):
    player_color = Color.black.value  # player using the env
    oponent_color = Color.green.value

    metadata = {"render.modes": ["log"]}

    _color_to_int = np.vectorize(lambda color: color.value)

    def __init__(
        self,
        shape,
        oponent,
        n_turn=300,
        allowed_time=120.0,
        boring_limit=200,
        render_mode="log",
    ):
        super(SeegaEnv, self).__init__()
        assert isinstance(
            oponent, Player
        ), "The oponent needs to be an instance of Player"

        self.nrow, self.ncol = self.board_shape = np.array(shape)
        self.oponent = oponent
        self.turn = 0
        self.n_turn = n_turn
        self.allowed_time = allowed_time
        self.just_stop = boring_limit
        self.done = False

        self.oponent_timer = Timer(
            "oponent_time", total_time=self.allowed_time, logger=None
        )

        self.observation_space = gym.spaces.Box(
            -1, 1, shape=self.board_shape, dtype=np.int8
        )  # -1 = black (player), 0 = empty, 1 = green (oponent)
        self.action_space = gym.spaces.Dict(
            {
                str(SeegaActionType.ADD.value): gym.spaces.Box(
                    0,
                    np.array([self.ncol, self.nrow]),
                    shape=np.array([2]),
                    dtype=np.uint8,
                ),
                str(SeegaActionType.MOVE.value): gym.spaces.Box(
                    0,
                    np.array([self.ncol, self.nrow, 3]),
                    shape=np.array([3]),
                    dtype=np.uint8,
                ),  # 0 = left, 1 = up, 2 = right, 3 = down
            }
        )

        self._reset()

    # def _log_timer(fun):
    #     def inner(self, *args, **kwargs):
    #         self.timer.start()
    #         results = fun(self, *args, **kwargs)
    #         self.timer.stop()
    #         return results

    #     return inner

    def _state_to_ndarray(self, state):
        return self._color_to_int(state.board.get_board_state())

    def _reset(self):
        self.done = False
        self.board = Board(self.board_shape)
        self.state = SeegaState(
            board=self.board, next_player=self.player_color, boring_limit=self.just_stop
        )
        # self.trace = Trace(
        #     self.state, players={-1: self.players[-1].name, 1: self.players[1].name}
        # )
        self.current_player = self.player_color

    def reset(self):
        self._reset()
        return self._state_to_ndarray(self.state)

    def step(self, action):
        self.turn += 1
        init_state = deepcopy(self.state)

        reward = 0.0

        # Our turn
        self.state.set_next_player(self.player_color)
        if self.state.phase == 1:
            actiontype = SeegaActionType.ADD
            to = tuple(action[str(actiontype.value)])
            action = SeegaAction(actiontype, to=to)
        elif self.state.phase == 2:
            actiontype = SeegaActionType.MOVE
            *at, move = action[str(actiontype.value)]
            dx = Action.get_dx(move)
            at = tuple(at)
            to = tuple(map(operator.add, at, dx))
            action = SeegaAction(actiontype, at=at, to=to)
        else:
            raise ValueError("state.phase not in 1, 2")

        result = SeegaRules.act(self.state, action, self.player_color)
        # Not legal move
        if isinstance(result, bool):
            reward += Reward.ILLEGAL_MOVE.value
        else:
            self.state, self.done = result

        # Oponent turn
        self.state.set_next_player(self.oponent_color)
        if not SeegaRules.is_player_stuck(self.state, self.oponent_color):
            oponent_remaining_time = self.oponent_timer.remain_time()
            self.oponent_timer.start()
            oponent_action = self.oponent.play(self.state, oponent_remaining_time)
            self.oponent_timer.stop()
            oponent_result = SeegaRules.act(
                self.state, oponent_action, self.oponent_color
            )
            if not isinstance(oponent_result, bool):
                self.state, self.done = oponent_result

        reward += Reward.PLAYER_PART.value * (
            init_state.score[self.player_color] - self.state.score[self.player_color]
        )
        reward += Reward.OPONENT_PART.value * (
            init_state.score[self.oponent_color] - self.state.score[self.oponent_color]
        )
        next_state = self._state_to_ndarray(self.state)

        done = self.done and self.turn < self.n_turn
        info = dict(**SeegaRules.get_results(self.state))  # TODO : Get the winner

        return next_state, reward, self.done, info

    def _game_step(self, action):
        """Plays one step of the game. Takes an action and perform in the environment.

        Args:
            action (Action): An action containing the move from a player.

        Returns:
            bool: Dependent on the validity of the action will return True if the was was performed False if not.
        """
        assert isinstance(
            action, SeegaAction
        ), "action has to be an Action class object"
        result = SeegaRules.act(self.state, action, self.current_player)
        if isinstance(result, bool):
            return False
        else:
            self.state, self.done = result
            self.current_player = self.state.get_next_player()
            return True

    @classmethod
    def as_TF(cls, *args, **kwargs):
        from tf_agents.environments import tf_py_environment
        from tf_agents.environments.gym_wrapper import GymWrapper 

        env = SeegaEnv(*args, **kwargs)
        suite_gym = GymWrapper(env)
        return tf_py_environment.TFPyEnvironment(suite_gym)

    def play_game(self):
        hit = 0

        timer_first_player = Timer(
            "first_player", total_time=self.allowed_time, logger=None
        )
        timer_second_player = Timer(
            "second_player", total_time=self.allowed_time, logger=None
        )
        turn = self.player_color
        while not self.done:
            hit += 1
            state = deepcopy(self.state)
            remain_time = (
                timer_first_player.remain_time()
                if turn == -1
                else timer_second_player.remain_time()
            )
            remain_time_copy = deepcopy(remain_time)
            if SeegaRules.is_player_stuck(state, turn):
                state.set_next_player(turn * -1)
                self.state.set_next_player(turn * -1)
                turn = turn * -1
                self.current_player = state.get_next_player()
            if remain_time > 0:
                timer_first_player.start() if turn == -1 else timer_second_player.start()
                action = self.players[turn].play(state, remain_time_copy)
                elapsed_time = (
                    timer_first_player.stop()
                    if turn == -1
                    else timer_second_player.stop()
                )
                remain_time = (
                    timer_first_player.remain_time()
                    if turn == -1
                    else timer_second_player.remain_time()
                )
                if self._game_step(action):
                    print(
                        "Action performed successfully by",
                        turn,
                        " in",
                        str(elapsed_time),
                        " rest ",
                        remain_time,
                    )
                else:
                    print("An illegal move were given. Performing a random move")
                    print(f"Lunching a random move for {turn}")
                    action = SeegaRules.random_play(
                        state, turn
                    )  # TODO: Should we use the original state?

            else:
                print("Not remain time for ", turn, " Performing a random move")
                print(f"Lunching a random move for {turn}")
                action = SeegaRules.random_play(
                    state, turn
                )  # TODO: Should we use the original state?
            # self.trace.add(self.state)
            self.players[turn].update_player_infos(self.get_player_info(turn))
            turn = self.state.get_next_player()
        print("\nIt's over.")

    def get_player_info(self, player):
        return self.state.get_player_info(player)

    def is_end_game(self):
        return self.done


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", help="total number of seconds credited to each player")
    parser.add_argument("-ai0", help="path to the ai that will play as player 0")
    parser.add_argument("-ai1", help="path to the ai that will play as player 1")
    args = parser.parse_args()

    # set the time to play
    allowed_time = float(args.t) if args.t is not None else 120.0

    player_type = ["human", "human"]
    player_type[0] = args.ai0 if args.ai0 != None else "human"
    player_type[1] = args.ai1 if args.ai1 != None else "human"
    for i in range(2):
        if player_type[i].endswith(".py"):
            player_type[i] = player_type[i][:-3]
    agents = {}

    # load the agents
    k = -1
    for i in range(2):
        if player_type[i] != "human":
            j = player_type[i].rfind("/")
            # extract the dir from the agent
            dir = player_type[i][:j]
            # add the dir to the system path
            sys.path.append(dir)
            # extract the agent filename
            file = player_type[i][j + 1 :]
            # create the agent instance
            agents[k] = getattr(__import__(file), "AI")(Color(k))
            k *= -1
    if None in agents:
        raise Exception(
            "Problems in  AI players instances. \n"
            "Usage:\n"
            "-t allowed time for each ai \n"
            "\t total number of seconds credited to each agent \n"
            "-ai0 ai0_file.py \n"
            "\t path to the ai that will play as player 0 \n"
            "-ai1 ai1_file.py\n"
            "\t path to the ai that will play as player 1 \n"
            "-s sleep time \n"
            "\t time(in second) to show the board(or move)"
        )
    game = SeegaEnv((5, 5), agents, allowed_time=allowed_time)
    game.play_game()
