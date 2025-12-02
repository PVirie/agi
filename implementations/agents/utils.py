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


def convert_chw_to_4bit(frame_input):
    """
    Converts a (C, H, W) array of integers (0-15) into 4-bit binary channels.
    
    Args:
        frame_input (np.array): Shape (C, H, W) or (H, W).
                                If (H, W) is provided, it treats C=1.
    
    Returns:
        np.array: Shape (C*4, H, W) as float32.
                  Expands every input channel into 4 binary output channels.
    """
    # 1. Standardize input to (C, H, W)
    # If input is (H, W), add the channel dimension at the start
    if frame_input.ndim == 2:
        frame = frame_input[np.newaxis, ...]
    else:
        frame = frame_input

    # 2. Prepare for unpacking
    # We reshape to (C, 1, H, W) so we can expand the '1' into '8' bits 
    # without mixing up the existing channels (C).
    frame_u8 = frame.astype(np.uint8)[:, np.newaxis, :, :]
    
    # 3. Unpack bits along axis 1
    # Shape becomes: (C, 8, H, W)
    bits = np.unpackbits(frame_u8, axis=1)
    
    # 4. Slice the last 4 bits (the lower nibble: 0000[XXXX])
    # Shape becomes: (C, 4, H, W)
    bits = bits[:, 4:, :, :]
    
    # 5. Merge the original channels (C) with the new bits (4)
    # Shape becomes: (C*4, H, W)
    # We use -1 to automatically calculate the new channel count
    output = bits.reshape(-1, frame.shape[-2], frame.shape[-1])
    
    return output.astype(np.float32)


def extract_frame(frame_data: FrameData) -> torch.Tensor:
    """Convert frame data to tensor format for the model."""
    # Convert frame to numpy array with color indices 0-15

    if frame_data.is_empty():
        # Return a zero tensor if frame is empty
        return frame_data.state, np.zeros((4*64*64), dtype=np.float32), frame_data.score

    frame = np.array(frame_data.frame, dtype=np.int64)  # shape (C, H, W)
    frame = convert_chw_to_4bit(frame)  # shape (4, H, W)
    frame = np.reshape(frame, (4*64*64))  # flatten to (4*64*64,)
    
    return frame_data.state, frame, frame_data.score
