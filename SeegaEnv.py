#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Apr 12, 13:48:50
@last modified : 2021 Apr 19, 14:44:13
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
from collections.abc import Iterable
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
            -1, 1, shape=self.board_shape
        )  # -1 = black (player), 0 = empty, 1 = green (oponent)
        # self.action_space = gym.spaces.Dict(
        #     {
        #         str(SeegaActionType.ADD.value): gym.spaces.Box(
        #             0,
        #             np.array([self.ncol, self.nrow]),
        #             shape=np.array([2]),
        #             dtype=np.uint8,
        #         ),
        #         str(SeegaActionType.MOVE.value): gym.spaces.Box(
        #             0,
        #             np.array([self.ncol, self.nrow, 3]),
        #             shape=np.array([3]),
        #             dtype=np.uint8,
        #         ),  # 0 = left, 1 = up, 2 = right, 3 = down
        #     }
        # )
        # self.action_space = gym.spaces.Box(
        #     0, np.array([self.ncol, self.nrow, 3]), shape=np.array([3]), dtype=np.uint8
        # )  # 0 = left, 1 = up, 2 = right, 3 = down
        self.action_space = gym.spaces.Discrete(self.ncol * self.nrow * 4)

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

    def _action_from_3D_to_1D(self, x: int, y: int, z: int):
        return (z * self.nrow * self.ncol) + (y * self.nrow) + x

    def _action_from_1D_to_3D(self, index: int):
        z = index // (self.nrow * self.ncol)
        index -= z * self.nrow * self.ncol
        y = index // self.nrow
        x = index % self.nrow
        return x, y, z

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
        self._fill_board()

    def _fill_board(self):
        turn = self.player_color
        while self.state.phase == 1:
            self.state.set_next_player(turn)
            action = SeegaRules.random_play(self.state, turn)
            result = SeegaRules.act(self.state, action, turn)

            if not isinstance(result, bool):
                self.state, _ = result

            turn *= -1

    def reset(self):
        self._reset()
        state = self._state_to_ndarray(self.state)
        return state

    def step(self, action):
        self.turn += 1
        init_state = deepcopy(self.state)

        reward = 0.0

        # Our turn
        self.state.set_next_player(self.player_color)
        if not isinstance(action, Iterable):
            action = self._action_from_1D_to_3D(action)

        *at, move = action
        dx = Action.get_dx(move)
        at = tuple(at)
        to = tuple(map(operator.add, at, dx))
        action = SeegaAction(SeegaActionType.MOVE, at=at, to=to)

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
            try:
                oponent_action = self.oponent.play(self.state, oponent_remaining_time)
            except Exception as e:
                print(e)
                oponent_action = None
            self.oponent_timer.stop()
            if isinstance(oponent_action, SeegaAction):
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
        info = dict(**SeegaRules.get_results(self.state)) 

        return next_state, reward, self.done, info

    @classmethod
    def as_TF(cls, *args, **kwargs):
        from tf_agents.environments import tf_py_environment
        from tf_agents.environments.gym_wrapper import GymWrapper

        env = SeegaEnv(*args, **kwargs)
        suite_gym = GymWrapper(env)
        return tf_py_environment.TFPyEnvironment(suite_gym)


class RaySeegaEnv(SeegaEnv):
    def __init__(self, env_config):
        super(RaySeegaEnv, self).__init__(**env_config)
