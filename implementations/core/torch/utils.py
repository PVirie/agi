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



def frame_to_tensor(self, frame_data: FrameData) -> torch.Tensor:
    """Convert frame data to tensor format for the model."""
    # Convert frame to numpy array with color indices 0-15
    frame = np.array(frame_data.frame, dtype=np.int64)
    
    # Take the last frame (in case of an animation of frames)
    frame = frame[-1]
    
    assert frame.shape == (self.grid_size, self.grid_size)
    
    # One-hot encode: (64, 64) -> (16, 64, 64)
    tensor = torch.zeros(self.num_colours, self.grid_size, self.grid_size, dtype=torch.float32)
    tensor.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
    
    return tensor.to(self.device)


def frame_data_to_tensors(frames: list[FrameData]) -> torch.Tensor:
    batch_size = len(frames)
    grid_size = 64
    num_channels = 16

    tensor_data = torch.zeros((batch_size, grid_size, grid_size, num_channels), dtype=torch.float32)
    for i, frame_data in enumerate(frames):
        for y in range(min(grid_size, len(frame_data.frame))):
            for x in range(min(grid_size, len(frame_data.frame[y]))):
                cell_value = frame_data.frame[y][x]
                if 0 <= cell_value < num_channels:
                    tensor_data[i, y, x, cell_value] = 1.0  # One-hot encoding
                    
    return tensor_data