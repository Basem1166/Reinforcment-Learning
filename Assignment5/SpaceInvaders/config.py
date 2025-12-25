class Config:
    ENV_NAME = "SpaceInvadersNoFrameskip-v4"
    GRAYSCALE = True
    FRAME_STACK = 3
    SCREEN_SIZE = 84
    RECORD_VIDEO = True
    VIDEO_DIR = "videos"

    TOTAL_EPISODES_WM = 50
    TOTAL_EPISODES_CONTROLLER = 50
    LR_WM = 1e-4
    LR_CONTROLLER = 1e-4

    LATENT_DIM = 128
    RNN_HIDDEN = 256
    CONTROLLER_HIDDEN = 128

    WAND_B_PROJECT_WM = "world-models-spaceinvaders"
    WAND_B_PROJECT_CONTROLLER = "controller-spaceinvaders"
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_FREQ = 10

    EVAL_EPISODES = 5
    DEVICE = "cuda"
