"""
Created on 26 oct. 12:02 2020

@author: HaroldKS
"""

from core.rules import Rule
from core import Color
from seega.seega_actions import SeegaActionType, SeegaAction


class SeegaRules(Rule):

    @staticmethod
    def is_legal_move(state, action, player):

        phase = state.phase
        action = action.get_action_as_dict()

        if state.get_next_player() == player:
            if phase == 1:
                if action['action_type'] == SeegaActionType.ADD and state.in_hand[player]:
                    empty_cells = state.get_board().get_all_empty_cells()
                    if empty_cells and action['action']['to'] in empty_cells \
                            and not state.get_board().is_center(action['action']['to']):
                        return True
            elif phase == 2:
                if SeegaRules.is_player_stuck(state, player * -1):
                    if action['action']['at'] in SeegaRules.get_unstucked_piece(state, player * -1):
                        return True
                    return False
                if state.get_next_player() == player:
                    if action['action_type'] == SeegaActionType.MOVE:
                        if state.get_board().get_cell_color(action['action']['at']) == Color(player):
                            effective_moves = SeegaRules.get_effective_cell_moves(state, action['action']['at'])
                            if effective_moves and action['action']['to'] in effective_moves:
                                return True
                    return False
            return False
        return False

    @staticmethod
    def get_effective_cell_moves(state, cell):
        """Give the effective(Only the possible ones) moves a player can make regarding a piece on the board.

        Args:
            state (YoteState): The current game state.
            cell ((int, int)): The coordinates of the piece on the board.
            player (int): The number of the player making the move.

        Returns:
            List: A list containing all the coordinates where the piece can go.
        """
        board = state.get_board()
        if board.is_cell_on_board(cell):
            possibles_moves = SeegaRules._get_rules_possibles_moves(cell, board.board_shape)
            effective_moves = []
            i, j = cell
            for move in possibles_moves:
                if board.is_empty_cell(move):
                    effective_moves.append(move)
            return effective_moves

    @staticmethod
    def _get_rules_possibles_moves(cell, board_shape):
        """Give all possibles moves for a piece according the game rules (Up, down, left, right).

        Args:
            cell ((int, int)): The coordinates of the piece on the board.
            board_shape ((int, int)): The board shape.

        Returns:
            List: A list containing all the coordinates where the piece could go.
        """
        return [(cell[0] + a[0], cell[1] + a[1])
                for a in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if ((0 <= cell[0] + a[0] < board_shape[0]) and (0 <= cell[1] + a[1] < board_shape[1]))]

    @staticmethod
    def act(state, action, player):  # TODO : Wondering if I should make the random move here.
        """Take the state and the player's action and make the move if possible.

        Args:
            state (YoteState): A state object from the yote game.
            action (Action): An action object containing the move.
            player (int): The number of the player making the move.
            rewarding_move (bool, optional): True if the move is a stealing move. Defaults to False.

        Returns:
            bool: True if everything goes fine and the move was made. False is else.
        """
        if SeegaRules.is_legal_move(state, action, player):
            return SeegaRules.make_move(state, action, player)
        else:
            return False

    @staticmethod
    def make_move(state, action, player):

        # they are in the state object
        """Transform the action of the player to a move. The move is made and the reward computed.

        Args:
            state (YoteState): A state object from the yote game.
            action (Action): An action object containing the move.
            player (int): The number of the player making the move.
            rewarding_move (bool, optional): True if the move is a stealing move. Defaults to False.

        Returns: (next_state, done, next_is_reward): Gives the next state of the game along with the game status and
        the type of the next step.
        """

        board = state.get_board()
        json_action = action.get_json_action()
        action = action.get_action_as_dict()
        phase = state.phase
        reward = 0

        if phase == 1 and action['action_type'] == SeegaActionType.ADD:
            state.in_hand[player] -= 1
            board.fill_cell(action['action']['to'], Color(player))

        elif phase == 2 and action['action_type'] == SeegaActionType.MOVE:
            at = action['action']['at']
            to = action['action']['to']
            captured_pieces = SeegaRules.captured(board, to, player)
            state.captured = captured_pieces
            board.empty_cell(at)
            board.fill_cell(to, Color(player))
            if captured_pieces:
                reward = len(captured_pieces)
                state.boring_moves = 0
                for piece in captured_pieces:
                    board.empty_cell(piece)
            else:
                state.boring_moves += 1

        state.set_board(board)
        state.score[player] += reward
        state.set_latest_player(player)
        state.set_latest_move(json_action)
        if phase == 1:
            if state.in_hand[player] == 0 and state.in_hand[player * -1] == 0:
                state.set_next_player(player)
                state.phase = 2
            elif state.in_hand[player] != 0 and state.in_hand[player] % 2 != 0:
                state.set_next_player(player)
            elif state.in_hand[player] % 2 == 0:
                state.set_next_player(player * -1)
        elif phase == 2:
            if SeegaRules.is_player_stuck(state, player * -1):
                state.set_next_player(player)
            else:
                state.set_next_player(player * -1)

        done = SeegaRules.is_end_game(state)
        return state, done

    @staticmethod
    def random_play(state, player):
        """Return a random move for a player at a given state.

        Args:
            state (YoteState): A state object from the yote game.
            player (int): The number of the player making the move.

        Returns:
            action : An action
        """
        import random
        # print("Player, ", player)
        actions = SeegaRules.get_player_all_cases_actions(state, player)
        choose = random.choice(actions)
        return choose

    @staticmethod
    def get_player_all_cases_actions(state, player):
        if state.phase == 2 and SeegaRules.is_player_stuck(state, player * -1):
            actions = list()
            piece_to_move = SeegaRules.get_unstucked_piece(state, player * -1)
            for piece in piece_to_move:
                moves = SeegaRules.get_effective_cell_moves(state, piece)
                if moves:
                    for move in moves:
                        actions.append(SeegaAction(action_type=SeegaActionType.MOVE, at=piece, to=move))
            return actions
        else:
            return SeegaRules.get_player_actions(state, player)



    @staticmethod
    def _get_opponent_neighbours(board, cell, player):

        possibles_neighbours = SeegaRules._get_rules_possibles_moves(cell, board.board_shape)
        enemies = list()
        for neighbour in possibles_neighbours:
            if board.get_cell_color(neighbour) == Color(player * -1):
                enemies.append(neighbour)
        return enemies

    @staticmethod
    def captured(board, move, player):
        opponent_pieces = SeegaRules._get_opponent_neighbours(board, move, player)
        captured_pieces = list()
        if opponent_pieces:
            for piece in opponent_pieces:
                if piece != (board.board_shape[0] // 2, board.board_shape[1] // 2):
                    if piece[0] == move[0]:
                        if piece[1] < move[1] and 0 <= piece[1] - 1 < board.board_shape[0] and \
                                board.get_cell_color((piece[0], piece[1] - 1)) == Color(player):
                            captured_pieces.append(piece)

                        if piece[1] > move[1] and 0 <= piece[1] + 1 < board.board_shape[0] and \
                                board.get_cell_color((piece[0], piece[1] + 1)) == Color(player):
                            captured_pieces.append(piece)
                    if piece[1] == move[1]:
                        if piece[0] < move[0] and 0 <= piece[0] - 1 < board.board_shape[0] and \
                                board.get_cell_color((piece[0] - 1, piece[1])) == Color(player):
                            captured_pieces.append(piece)
                        if piece[0] > move[0] and 0 <= piece[0] + 1 < board.board_shape[0] and \
                                board.get_cell_color((piece[0] + 1, piece[1])) == Color(player):
                            captured_pieces.append(piece)

        return captured_pieces

    @staticmethod
    def get_player_actions(state, player):
        """Provide for a player and at a state all of his possible actions.

        Args:
            state (YoteState): A state object from the yote game.
            player (int, optional): True if the move is a stealing move. Defaults to False.

        Returns:
            List[YoteAction]: Contains all possible actions for a player at the given state.
        """

        actions = []
        phase = state.phase
        board = state.get_board()

        if phase == 1:
            empty_cells = board.get_all_empty_cells_without_center()
            if empty_cells and state.in_hand[player]:
                for cell in empty_cells:
                    actions.append(SeegaAction(action_type=SeegaActionType.ADD, to=cell))
            return actions
        elif phase == 2:
            player_pieces = board.get_player_pieces_on_board(Color(player))
            for piece in player_pieces:
                moves = SeegaRules.get_effective_cell_moves(state, piece)
                if moves:
                    for move in moves:
                        actions.append(SeegaAction(action_type=SeegaActionType.MOVE, at=piece, to=move))
            return actions

    @staticmethod
    def is_player_stuck(state, player):  # WARNING: Note used yet
        """Check if a player has the possibility to make a move

        Args:
            state (YoteState): A state object from the yote game.
            player (int): The number of the player making the move.

        Returns:
            bool: True if a player can make a move. False if not.
        """
        return len(SeegaRules.get_player_actions(state, player)) == 0

    @staticmethod
    def is_end_game(state):
        """Check if the given state is the last one for the current game.

        Args:
            state (YoteState): A state object from the yote game.

        Returns:
            bool: True if the given state is the final. False if not.
        """
        if state.phase == 1:
            return False
        if SeegaRules.is_boring(state):
            return True
        latest_player_score = state.score[state.get_latest_player()]
        if latest_player_score >= state.MAX_SCORE:
            return True
        return False

    @staticmethod
    def is_boring(state):
        """Check if the game is ongoing without winning moves

        Args:
            state (YoteState): A state object from the yote game.
        Returns:
            bool: True if the game is boring. False if else.
        """
        return state.boring_moves >= state.just_stop

    @staticmethod
    def get_results(state):  # TODO: Add equality case.
        """Provide the results at the end of the game.

        Args:
            state (YoteState): A state object from the yote game.

        Returns:
            Dictionary: Containing the winner and the game score.
        """
        tie = False
        if state.score[-1] == state.score[1]:
            tie = True
        return {'tie': tie, 'winner': max(state.score, key=state.score.get),
                'score': state.score}

    @staticmethod
    def get_unstucked_piece(state, player):
        board = state.get_board()
        player_pieces = board.get_player_pieces_on_board(Color(player))
        piece_to_move = set()
        for piece in player_pieces:
            opponent_pieces = SeegaRules._get_opponent_neighbours(board, piece, player)
            if len(opponent_pieces) > 0:
                for opp_piece in opponent_pieces:
                    if len(SeegaRules.get_effective_cell_moves(state, opp_piece)) > 0:
                        piece_to_move.add(opp_piece)
        return list(piece_to_move)





