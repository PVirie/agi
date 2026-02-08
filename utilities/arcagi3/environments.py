import os
import requests
from enum import Enum
from typing import List
import logging
from .utils import extract_frame, check_frame_difference


API_KEY = os.getenv("ARC_API_KEY", "")
HOST = os.getenv("HOST", "three.arcprize.org")
PORT = os.getenv("PORT", "443")
SCHEME = os.getenv("SCHEME", "https")

base_url = f"{SCHEME}://{HOST}:{PORT}"

list_games_endpoint = "/api/games"
score_card_open_endpoint = "/api/scorecard/open"
score_card_close_endpoint = "/api/scorecard/close"
cmd_format_endpoint = "/api/cmd/{cmd}"

class Action_Type(Enum):
    """
        Available Actions:
            Action 0: Restart the current level, issue twice to reset the game from the first level. 
            This should not be used as a player action; lest the agent exploits only first level to accumulate score.
            I therefore split it into two actions.
            Action 1: Move up or Select A
            Action 2: Move down or Select B
            Action 3: Move left or Select C
            Action 4: Move right or Select D
            Action 5: Use, Interact, or Select E
            Action 6: This one is special, it takes additional two parameters:
                X coordinate (0-63)
                Y coordinate (0-63)
            Action 7: Undo
    """
    RESTART = 8
    RESET = 7
    A1 = 0
    A2 = 1
    A3 = 2
    A4 = 3
    A5 = 4
    A6 = 5
    A7 = 6

    def __repr__(self):
        return f"{self.name}"


action_type_to_str = {
    Action_Type.A1: "ACTION1",
    Action_Type.A2: "ACTION2",
    Action_Type.A3: "ACTION3",
    Action_Type.A4: "ACTION4",
    Action_Type.A5: "ACTION5",
    Action_Type.A6: "ACTION6",
    Action_Type.A7: "ACTION7"
}

return_action_to_action_type = {
    1: Action_Type.A1,
    2: Action_Type.A2,
    3: Action_Type.A3,
    4: Action_Type.A4,
    5: Action_Type.A5,
    6: Action_Type.A6,
}

class Game_State_Type(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    GAME_OVER = "GAME_OVER"
    WIN = "WIN"
    NOT_FINISHED = "NOT_FINISHED"
    IDLE = "IDLE"
    TRUNCATED = "TRUNCATED"
    RESET = "RESET"


class Game_State:
    def __init__(self, frame, state: str, score: int, delta_score: int, win_score: int, next_available_actions: List):
        self.frame = frame
        self.state = Game_State_Type(state)
        self.score = score
        self.win_score = win_score
        self.next_available_actions = [Action_Type(action) for action in next_available_actions]
        self.delta_score = delta_score
        self.matched_relative_index = -1  # relative index in frame history of last different frame

    def short_str(self):
        return f"{self.state.value}|{self.score}/{self.win_score}"
    
    def set_matched_relative_index(self, index: int):
        self.matched_relative_index = index
    

class Frame_History:
    def __init__(self, max_length: int):
        self.max_length = max_length
        # only store frame hash
        self.frame_hashes = []

    def clear(self):
        self.frame_hashes = []

    def add_frame(self, frame):
        frame_hash = hash(frame.tobytes())
        self.frame_hashes.append(frame_hash)
        if len(self.frame_hashes) > self.max_length:
            self.frame_hashes.pop(0)

    def find_matching_relative_index(self, frame) -> int:
        """
        return the relative index of the matching frame in history, 0 is the most recent frame
        return -1 if not found
        """
        frame_hash = hash(frame.tobytes())
        for i, prev_frame_hash in enumerate(reversed(self.frame_hashes)):
            if frame_hash == prev_frame_hash:
                return i
        return -1


class ARCAGI3_Remote_Environment:

    def __init__(self):
        self.request_session = requests.Session()
        self.request_session.headers.update({"X-API-Key": API_KEY, "Content-Type": "application/json"})

        self.all_game_metadata = None
        self.score_card_id = None

        self.is_started = False
        self.selected_game_ids = None
        self.guids = None

        self.return_states = None
        self.frame_histories = None


    async def list_games(self):
        url = base_url + list_games_endpoint
        response = self.request_session.get(url)
        response.raise_for_status()
        """
        [
            {
                "game_id": "ls20-016295f7601e",
                "title": "LS20"
            },
            {
                "game_id": "ft09-16726c5b26ff",
                "title": "FT09"
            }
        ]
        """
        results = response.json()
        known_public_game_titles = set(["LS20", "FT09", "VC33"])
        # add game type property to each game metadata
        for game_meta in results:
            if game_meta["title"] in known_public_game_titles:
                game_meta["game_type"] = "public"

        self.all_game_metadata = results
        return results
    

    async def start(self, selected_game_ids: List[str]):
        if self.is_started:
            return None
        
        # first get score card id, fire post request
        response = self.request_session.post(base_url + score_card_open_endpoint,
            json={}
        )
        response.raise_for_status()
        self.score_card_id = response.json().get("card_id", None)

        self.selected_game_ids = selected_game_ids
        self.guids = [None for _ in selected_game_ids]

        # set flag
        self.is_started = True
        self.return_states = [
            Game_State(frame=None, state=Game_State_Type.NOT_STARTED.value, score=0, delta_score=0, win_score=0, next_available_actions=[]) for _ in selected_game_ids
        ]
        self.frame_histories = [
            Frame_History(max_length=4) for _ in selected_game_ids
        ]
    

    async def close(self):
        if not self.is_started:
            return None
        
        # close score card
        url = base_url + score_card_close_endpoint
        payload = {"card_id": self.score_card_id}
        response = self.request_session.post(url, json=payload)
        response.raise_for_status()

        # set flag
        self.is_started = False
        self.return_states = None

        # return report
        return response.json()
    

    def reset_item(self, i, action_type):
        response = self.request_session.post(
            base_url + cmd_format_endpoint.format(cmd="RESET"),
            json={
                "card_id": self.score_card_id,
                "game_id": self.selected_game_ids[i],
                "guid": self.guids[i] if action_type != Action_Type.RESET else None
            }
        )
        response.raise_for_status()
        response_json = response.json()
        
        game_state = Game_State(
            frame=extract_frame(response_json.get("frame", None)),
            state=Game_State_Type.RESET if action_type == Action_Type.RESET else Game_State_Type.TRUNCATED,
            score=response_json.get("score", 0),
            delta_score=0,
            win_score=response_json.get("win_score", 0),
            next_available_actions=[return_action_to_action_type[aa] for aa in response_json.get("available_actions", [])]
        )
    
        # clear frame history
        self.guids[i] = response_json.get("guid", None)
        self.frame_histories[i].clear()

        return game_state


    def act_item(self, i, action_type: Action_Type, x: int = None, y: int = None):
        json_payload = {
            "card_id": self.score_card_id,
            "game_id": self.selected_game_ids[i],
            "guid": self.guids[i]
        }

        if action_type == Action_Type.A6:
            json_payload["x"] = x
            json_payload["y"] = y

        response = self.request_session.post(
            base_url + cmd_format_endpoint.format(cmd=action_type_to_str[action_type]),
            json=json_payload
        )
        response.raise_for_status()
        response_json = response.json()

        game_state = Game_State(
            frame=extract_frame(response_json.get("frame", None)),
            state=response_json.get("state", None),
            score=response_json.get("score", 0),
            delta_score=0,
            win_score=response_json.get("win_score", 0),
            next_available_actions=[return_action_to_action_type[aa] for aa in response_json.get("available_actions", [])]
        )
        game_state.delta_score = game_state.score - (self.return_states[i].score if self.return_states[i] is not None else 0)

        return game_state


    async def execute(self, actions):
        """
        actions is a list of tuple [(action_type: Action_Type, x: int, y: int)]
        allowing executing multiple games in one call
        if the i-th action is None, skip executing that game and return its last state
        """
        if not self.is_started:
            raise Exception("Environment not started. Call start() before execute().")

        something_changed = False
        for i, at in enumerate(actions):
            if at is None:
                self.return_states[i].state = Game_State_Type.IDLE
                self.return_states[i].delta_score = 0  # no change
                continue

            action_type = at[0]
            if action_type == Action_Type.RESET or action_type == Action_Type.RESTART:
                game_state = self.reset_item(i, action_type)
                game_state.delta_score = 0
                self.return_states[i] = game_state

            else:
                try:
                    game_state = self.act_item(i, action_type, x=at[1] if len(at) > 1 else None, y=at[2] if len(at) > 2 else None)
                    self.return_states[i] = game_state

                    # if WIN or GAME_OVER, need to reset the game
                    if game_state.state in [Game_State_Type.WIN.value, Game_State_Type.GAME_OVER.value]:
                        logging.info(f"Game {self.selected_game_ids[i]} ended with state {game_state.state}. Resetting...")
                        reset_game_state = self.reset_item(i, Action_Type.RESET)
                        reset_game_state.state = game_state.state # preserve the end state
                        reset_game_state.delta_score = game_state.delta_score # preserve the delta score
                        self.return_states[i] = reset_game_state

                except Exception as e:
                    logging.warning(f"Error executing action {action_type} for game {self.selected_game_ids[i]}: {e}")
                    logging.info(f"Reset and set state to TRUNCATED.")
                    game_state = self.reset_item(i, Action_Type.RESTART)
                    game_state.delta_score = 0
                    self.return_states[i] = game_state
            
            # update frame history
            if self.return_states[i].frame is not None:
                matched_index = self.frame_histories[i].find_matching_relative_index(self.return_states[i].frame)
                self.return_states[i].set_matched_relative_index(matched_index)
                self.frame_histories[i].add_frame(self.return_states[i].frame)

            # check if something changed
            if self.return_states[i].delta_score != 0 or self.return_states[i].state in [Game_State_Type.WIN, Game_State_Type.GAME_OVER]:
                something_changed = True

        
        return something_changed, self.return_states
    
