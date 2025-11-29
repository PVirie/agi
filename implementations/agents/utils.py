from agents.structs import FrameData, GameAction, GameState # Make sure to change from `..` imports
import torch
import numpy as np

# class FrameData(BaseModel):
#     game_id: str = ""
#     frame: list[list[list[int]]] = []
#     state: GameState = GameState.NOT_PLAYED
#     score: int = Field(0, ge=0, le=254)
#     action_input: ActionInput = Field(default_factory=lambda: ActionInput())
#     guid: Optional[str] = None
#     full_reset: bool = False
#     available_actions: list[GameAction] = Field(default_factory=list)

#     def is_empty(self) -> bool:
#         return len(self.frame) == 0

# Grid Structure
# Dimensions: Maximum 64x64 grid size
# Cell Values: Integer values 0-15 representing different states/colors
# Coordinate System: (0,0) at top-left, (x,y) format

# class GameState(str, Enum):
#     NOT_PLAYED = "NOT_PLAYED"
#     NOT_FINISHED = "NOT_FINISHED"
#     WIN = "WIN"
#     GAME_OVER = "GAME_OVER"


def extract_frame(frame_data: FrameData) -> torch.Tensor:
    """Convert frame data to tensor format for the model."""
    # Convert frame to numpy array with color indices 0-15

    if frame_data.is_empty():
        # Return a zero tensor if frame is empty
        return frame_data.state, np.zeros((64, 64), dtype=np.int64), frame_data.score

    frame = np.array(frame_data.frame, dtype=np.int64)
    
    # Take the last frame (in case of an animation of frames)
    frame = frame[-1]
    
    return frame_data.state, frame, frame_data.score
