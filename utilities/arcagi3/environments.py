import os
import requests
from enum import Enum
from typing import List
import logging

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


action_type_to_str = {
    Action_Type.A1: "ACTION1",
    Action_Type.A2: "ACTION2",
    Action_Type.A3: "ACTION3",
    Action_Type.A4: "ACTION4",
    Action_Type.A5: "ACTION5",
    Action_Type.A6: "ACTION6",
    Action_Type.A7: "ACTION7"
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
    def __init__(self, frame, state: str, score: int, win_score: int, next_available_actions: List):
        self.frame = frame
        self.state = Game_State_Type(state)
        self.score = score
        self.win_score = win_score
        self.next_available_actions = [Action_Type(action) for action in next_available_actions]


class ARCAGI3_Environment:

    def __init__(self):
        self.request_session = requests.Session()
        self.request_session.headers.update({"X-API-Key": API_KEY, "Content-Type": "application/json"})

        self.all_game_metadata = None
        self.score_card_id = None

        self.is_started = False
        self.selected_game_ids = None
        self.guids = None

        self.return_states = None


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
            Game_State(frame=None, state=Game_State_Type.NOT_STARTED.value, score=0, win_score=0, next_available_actions=[]) for _ in selected_game_ids
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
    

    async def execute(self, actions):
        """
        actions is a list of tuple [(action_type: Action_Type, x: int, y: int)]
        allowing executing multiple games in one call
        if the i-th action is None, skip executing that game and return its last state
        """
        if not self.is_started:
            raise Exception("Environment not started. Call start() before execute().")

        for i, at in enumerate(actions):
            if at is None:
                self.return_states[i].state = Game_State_Type.IDLE
                continue
            
            action_type = Action_Type(at[0])
            if action_type == Action_Type.RESET or action_type == Action_Type.RESTART:
                response = self.request_session.post(
                    base_url + cmd_format_endpoint.format(cmd="RESET"),
                    json={
                        "card_id": self.score_card_id,
                        "game_id": self.selected_game_ids[i],
                        "guid": self.guids[i] if action_type != Action_Type.RESET else None
                    }
                )
                response_json = response.json()
                self.guids[i] = response_json.get("guid", None)
                response_json["state"] = Game_State_Type.RESET if action_type == Action_Type.RESET else Game_State_Type.TRUNCATED
            else:
                json_payload = {
                    "card_id": self.score_card_id,
                    "game_id": self.selected_game_ids[i],
                    "guid": self.guids[i]
                }

                if action_type == Action_Type.A6:
                    x = at[1] if len(at) > 1 else None
                    y = at[2] if len(at) > 2 else None
                    json_payload["x"] = x
                    json_payload["y"] = y

                try:
                    response = self.request_session.post(
                        base_url + cmd_format_endpoint.format(cmd=action_type_to_str[action_type]),
                        json=json_payload
                    )
                    response.raise_for_status()
                    response_json = response.json()
                except Exception as e:
                    logging.warning(f"Error executing action {action_type} for game {self.selected_game_ids[i]}: {e}")
                    logging.info(f"Reset and set state to TRUNCATED.")
                    response = self.request_session.post(
                        base_url + cmd_format_endpoint.format(cmd="RESET"),
                        json={
                            "card_id": self.score_card_id,
                            "game_id": self.selected_game_ids[i],
                            "guid": self.guids[i]
                        }
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    response_json["state"] = Game_State_Type.TRUNCATED
                    continue

            game_state = Game_State(
                frame=response_json.get("frame", None),
                state=response_json.get("state", None),
                score=response_json.get("score", 0),
                win_score=response_json.get("win_score", 0),
                next_available_actions=response_json.get("next_available_actions", [])
            )
            self.return_states[i] = game_state
        
        return self.return_states
    
