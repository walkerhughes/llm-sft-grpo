# -*- coding: utf-8 -*-
"""
Original file is located at, thanks to OpenPipe!
    https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb
"""
import os
import random
import time
import math
import xml.etree.ElementTree as ET
from typing import TypedDict
from typing import Literal

# Third-party imports - unsloth must be imported first
from unsloth import FastLanguageModel
import numpy as np
import art
from art.local import LocalBackend
from dotenv import load_dotenv
import openai
from pydantic import BaseModel
import torch
from openpipe.client import OpenPipe
from huggingface_hub import HfApi  # For pushing to HF Hub

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
PROJECT_TITLE = BASE_MODEL.split("/")[1]

# Optional
WANDB_API_KEY = ""
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# Optional
OPENPIPE_API_KEY = ""
if OPENPIPE_API_KEY:
    os.environ["OPENPIPE_API_KEY"] = OPENPIPE_API_KEY

# Get Hugging Face token from environment
HF_TOKEN = os.environ.get("HF_TOKEN", "")


class TicTacToeGame(TypedDict):
    board: list[list[str]]
    agent_symbol: Literal["x", "o"]
    opponent_symbol: Literal["x", "o"]


def generate_game(board_length: int = 3) -> TicTacToeGame:
    board = [["_" for _ in range(board_length)] for _ in range(board_length)]
    agent_symbol = random.choice(["x", "o"])
    opponent_symbol = "x" if agent_symbol == "o" else "o"
    return {
        "board": board,
        "agent_symbol": agent_symbol,
        "opponent_symbol": opponent_symbol,
    }


def render_board(game: TicTacToeGame) -> str:
    board = game["board"]
    board_length = len(board)
    # print something like this:
    #    1   2   3
    # A  _ | x | x
    # B  o | _ | _
    # C  _ | o | _
    # where _ is an empty cell

    board_str = "   " + "   ".join([str(i + 1) for i in range(board_length)]) + "\n"
    for i in range(board_length):
        board_str += f"{chr(65 + i)}  {board[i][0]} | {board[i][1]} | {board[i][2]}\n"
    return board_str


def get_opponent_move(game: TicTacToeGame) -> tuple[int, int]:
    # get a random empty cell
    empty_cells = [
        (i, j) for i in range(3) for j in range(3) if game["board"][i][j] == "_"
    ]
    return random.choice(empty_cells)


def apply_agent_move(game: TicTacToeGame, move: str) -> None:
    board_length = len(game["board"])

    try:
        root = ET.fromstring(move)
        square = root.text
    except Exception:
        raise ValueError("Invalid xml")

    try:
        row_index = ord(square[0]) - 65
        col_index = int(square[1]) - 1
    except Exception as e:
        print(e)
        raise ValueError("Unable to parse square")

    if (
        row_index < 0
        or row_index >= board_length
        or col_index < 0
        or col_index >= board_length
    ):
        raise ValueError(
            f"Invalid move, row or column out of bounds: {row_index}, {col_index}"
        )

    # check if the move is valid
    if game["board"][row_index][col_index] != "_":
        raise ValueError("Square already occupied")

    game["board"][row_index][col_index] = game["agent_symbol"]


def check_winner(board: list[list[str]]) -> Literal["x", "o", "draw", None]:
    board_length = len(board)
    # check rows
    for row in board:
        if row.count(row[0]) == board_length and row[0] != "_":
            return row[0]
    # check columns
    for col in range(board_length):
        if [board[row][col] for row in range(board_length)].count(
            board[0][col]
        ) == board_length and board[0][col] != "_":
            return board[0][col]

    # top right to bottom left
    upward_diagonal = [board[i][board_length - i - 1] for i in range(board_length)]
    if (
        upward_diagonal.count(upward_diagonal[0]) == board_length
        and upward_diagonal[0] != "_"
    ):
        return upward_diagonal[0]

    # top left to bottom right
    downward_diagonal = [board[i][i] for i in range(board_length)]
    if (
        downward_diagonal.count(downward_diagonal[0]) == board_length
        and downward_diagonal[0] != "_"
    ):
        return downward_diagonal[0]

    # check for draw
    if all(cell != "_" for row in board for cell in row):
        return "draw"
    return None

# Initialize variables at module level but don't run async code
load_dotenv()
op_client = OpenPipe()
random.seed(42)
# Create backend object but don't register model yet
backend = LocalBackend(path="./.art")

# Moved the model registration into the main function
# No more top-level await statements

class TicTacToeScenario(BaseModel):
    step: int

@art.retry(exceptions=(openai.LengthFinishReasonError,))
async def rollout(
    model: art.Model, scenario: TicTacToeScenario
) -> art.Trajectory:
    game = generate_game()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": f"You are a tic-tac-toe player. You are playing against an opponent. Always choose the move most likely to lead to an eventual win. Return your move as an XML object with a single property 'move', like so: <move>A1</move>. Optional moves are 'A1', 'B3', 'C2', etc. You are the {game['agent_symbol']} symbol.",
            }
        ],
        reward=0,
    )

    move_number = 0

    if game["agent_symbol"] == "o":
        starting_opponent_move = get_opponent_move(game)
        game["board"][starting_opponent_move[0]][starting_opponent_move[1]] = game[
            "opponent_symbol"
        ]

    while check_winner(game["board"]) is None:
        trajectory.messages_and_choices.append(
            {"role": "user", "content": render_board(game)}
        )

        requested_at = int(time.time() * 1000)
        messages = trajectory.messages()

        try:
            client = model.openai_client()
            chat_completion = await client.chat.completions.create(
                model=model.get_inference_name(),
                messages=messages,
                max_completion_tokens=128,
            )
            last_completion = chat_completion
        except openai.LengthFinishReasonError as e:
            raise e
        except Exception as e:
            print("caught exception generating chat completion")
            print(e)
            # Define failing_trajectory at a function level to avoid global issues
            failing_trajectory = trajectory
            raise e

        try:
            if op_client.api_key:
                op_client.report(
                    requested_at=requested_at,
                    received_at=int(time.time() * 1000),
                    req_payload={
                        "model": model.name,
                        "messages": messages,
                        "metadata": {
                            "notebook-id": "tic-tac-toe",
                            "step": str(scenario.step),
                            "move_number": str(move_number),
                        },
                    },
                    resp_payload=chat_completion,
                    status_code=200,
                )
        except Exception as e:
            print(f"Error reporting to OpenPipe: {e}")

        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)

        try:
            apply_agent_move(game, content)
        except ValueError:
            trajectory.reward = -1 + (math.log(move_number + 1) / math.log(100))
            break

        move_number += 1
        if check_winner(game["board"]) is not None:
            break

        opponent_move = get_opponent_move(game)
        game["board"][opponent_move[0]][opponent_move[1]] = game["opponent_symbol"]

    winner = check_winner(game["board"])

    if winner == game["agent_symbol"]:
        trajectory.reward = 1
        trajectory.metrics["win"] = 1
    elif winner == game["opponent_symbol"]:
        trajectory.reward = 0
        trajectory.metrics["win"] = 0
    elif winner == "draw":
        trajectory.reward = 0.5
        trajectory.metrics["win"] = 0.5

    trajectory.metrics["num_moves"] = move_number

    if op_client.api_key:
        try:
            reported_win = (
                trajectory.metrics["win"] if "win" in trajectory.metrics else -1
            )
            op_client.report(
                requested_at=requested_at,
                received_at=int(time.time() * 1000),
                req_payload={
                    "model": model.name,
                    "messages": messages,
                    "metadata": {
                        "notebook-id": "tic-tac-toe",
                        "step": str(scenario.step),
                        "num_moves": str(move_number),
                        "win": str(reported_win),
                        "reward": str(trajectory.reward),
                    },
                },
                resp_payload=chat_completion,
                status_code=200,
            )
        except Exception as e:
            print(f"Error reporting to OpenPipe: {e}")

    return trajectory


def play_demo_game(model_path: str):
    """Play a demonstration game with the trained model"""
    import torch
    from unsloth import FastLanguageModel
    
    peft_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=16384,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(peft_model)
    
    # Play a game with the trained model
    game = generate_game()
    move_number = 0
    
    messages = [
        {
            "role": "system",
            "content": f"You are a tic-tac-toe player. You are playing against an opponent. Always choose the move most likely to lead to an eventual win. Return your move as an XML object with a single property 'move', like so: <move>A1</move>. Optional moves are 'A1', 'B3', 'C2', etc. You are the {game['agent_symbol']} symbol.",
        },
    ]
    
    if game["agent_symbol"] == "o":
        starting_opponent_move = get_opponent_move(game)
        game["board"][starting_opponent_move[0]][starting_opponent_move[1]] = game[
            "opponent_symbol"
        ]
    
    while check_winner(game["board"]) is None:
        rendered_board = render_board(game)
        messages.append({"role": "user", "content": rendered_board})
        
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to("cuda")
        
        content = ""
        
        try:
            content = get_completion(peft_model, tokenizer, inputs)
        except Exception as e:
            print("caught exception generating chat completion", e)
            raise e
        
        messages.append({"role": "assistant", "content": content})
        
        try:
            apply_agent_move(game, content)
            move_number += 1
        except ValueError:
            raise ValueError(f"Invalid move on move {move_number}: {content}")
        
        # print the board every move
        print(f"\nmove {move_number}")
        print(f"board:\n{rendered_board}")
        print(f"agent move: {content}")
        print(f"updated board:\n{render_board(game)}")
        
        if check_winner(game["board"]) is not None:
            break
        move_number += 1
        
        opponent_move = get_opponent_move(game)
        game["board"][opponent_move[0]][opponent_move[1]] = game["opponent_symbol"]
    
    winner = check_winner(game["board"])
    
    print(f"game finished in {move_number} moves")
    
    if winner == game["agent_symbol"]:
        print("game won! 💪")
    elif winner == game["opponent_symbol"]:
        print("game lost! 😢")
    elif winner == "draw":
        print("draw! 🤷‍♂️")
    
    print(f"final board:\n\n{render_board(game)}")
    
    # Don't return any complex objects


async def main(base_model: str = BASE_MODEL, epochs: int = 50, num_rollouts: int = 48, hf_repo_id: str = None, push_to_hub: bool = True, run_demo: bool = False):
    """
    Main function to run the tic-tac-toe training and evaluation.
    
    Args:
        base_model: The base model to use for training (default: BASE_MODEL)
        epochs: Number of training epochs (default: 5)
        num_rollouts: Number of rollouts per epoch (default: 5)
        hf_repo_id: Hugging Face repository ID to push the model to (default: username/model-name)
        push_to_hub: Whether to push the model to Hugging Face Hub (default: True)
        run_demo: Whether to run a demo game after training (default: False)
        
    Returns:
        dict: Information about the trained model
    """
    # Set up project title based on model name
    project_title = base_model.split("/")[1]
    
    # Initialize OpenPipe client and backend
    # These are already initialized at the module level, so we'll reuse them
    
    # Create and register the model
    model = art.TrainableModel(
        name=project_title, 
        project=project_title, 
        base_model=base_model
    )
    await model.register(backend)
    
    # Training loop
    for i in range(await model.get_step(), epochs):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, TicTacToeScenario(step=i)) for _ in range(num_rollouts)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
        )
        await model.delete_checkpoints()
        await model.train(train_groups, config=art.TrainConfig(learning_rate=1e-4))
    
    # Calculate the path to the trained model
    model_step = await model.get_step()
    lora_model_path = f".art/{model.project}/models/{model.name}/{model_step:04d}"
    print(f"Model saved to {lora_model_path}\n")
    
    # Play a game with the trained model to validate it (optionally)
    if run_demo:
        try:
            print("Starting demo game with the trained model...")
            play_demo_game(lora_model_path)
            print("Demo game completed successfully!")
        except Exception as e:
            print(f"Error in demo game: {e}")
            print("Continuing with model upload despite demo game error...")
    
    # Push to HuggingFace Hub if requested
    hf_repo_info = None
    if push_to_hub and HF_TOKEN:
        try:
            hf_repo_info = push_model_to_hub(lora_model_path, hf_repo_id, base_model)
            print(f"Model pushed to Hugging Face Hub: {hf_repo_info['repo_url']}")
        except Exception as e:
            print(f"Error pushing to Hugging Face Hub: {e}")
            # Return information even if upload failed
            return {
                "local_path": lora_model_path,
                "base_model": base_model,
                "project": model.project,
                "model_name": model.name,
                "step": model_step,
                "huggingface_error": str(e)
            }
    
    # Return information about the model
    return {
        "local_path": lora_model_path,
        "base_model": base_model,
        "project": model.project,
        "model_name": model.name,
        "step": model_step,
        "huggingface_repo": hf_repo_info
    }

def push_model_to_hub(model_path, repo_id=None, base_model_name=None):
    """
    Push a trained model to Hugging Face Hub
    
    Args:
        model_path: Path to the model directory
        repo_id: Repository ID (required, e.g. "banachspace/Qwen-2.5-0.5B-Instruct-TicTacToe-GRPO")
        base_model_name: Name of the base model (for documentation)
        
    Returns:
        dict: Information about the pushed model repository
    """
    import os
    import shutil
    import tempfile
    from huggingface_hub import HfApi, create_repo, upload_folder
    
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set. Please set it to push to Hugging Face Hub.")
    
    if not repo_id:
        raise ValueError("repo_id must be specified (e.g. 'banachspace/Qwen-2.5-0.5B-Instruct-TicTacToe-GRPO')")
        
    # Initialize the Hugging Face API client
    api = HfApi(token=HF_TOKEN)
    
    print(f"Will push model to: {repo_id}")
    
    # Create the repository if it doesn't exist
    try:
        repo_info = api.repo_info(repo_id)
        print(f"Repository {repo_id} already exists")
    except Exception:
        print(f"Creating new repository: {repo_id}")
        repo_url = create_repo(repo_id, token=HF_TOKEN, private=False, exist_ok=True)
        print(f"Created repository: {repo_url}")
    
    # Create the README content without using format() to avoid issues with JSON brackets
    readme_content = f"""# Tic-Tac-Toe Reinforcement Learning Model

This model was trained to play Tic-Tac-Toe using reinforcement learning through ART (AI Reinforcement Training).

## Model Details
- **Base Model**: {base_model_name}
- **Training Method**: Reinforcement Learning with Human Feedback (RLHF)
- **Task**: Playing Tic-Tac-Toe optimally

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Example usage for Tic-Tac-Toe
messages = [
    {{"role": "system", "content": "You are a tic-tac-toe player. Return your move as an XML object like <move>A1</move>. You are the x symbol."}},
    {{"role": "user", "content": "   1   2   3\\nA  _ | _ | _\\nB  _ | _ | _\\nC  _ | _ | _"}}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"AI move: {{response}}")
```

## Training Process
The model was trained using generative reinforcement learning on the task of playing Tic-Tac-Toe optimally.

- Training framework: ART (AI Reinforcement Training)
- Reward function: Win = 1.0, Draw = 0.5, Loss = 0.0
- Learning method: Group Relative Policy Optimization (GRPO)
"""
    
    # Create a README file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        readme_path = f.name
        f.write(readme_content)
    
    # Create a temporary directory to hold the files
    temp_dir = tempfile.mkdtemp()
    try:
        # Copy the model files to the temporary directory
        for item in os.listdir(model_path):
            s = os.path.join(model_path, item)
            d = os.path.join(temp_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        
        # Add the README file
        shutil.copy2(readme_path, os.path.join(temp_dir, "README.md"))
        
        # Upload the folder to Hugging Face Hub
        upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN
        )
        
        # Return information about the repository
        return {
            "repo_id": repo_id,
            "repo_url": f"https://huggingface.co/{repo_id}",
            "username": repo_id.split('/')[0]
        }
    
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        os.unlink(readme_path)

# Extract get_completion function that was previously nested
def get_completion(peft_model, tokenizer, inputs) -> str:
    with torch.no_grad():
        outputs = peft_model.generate(
            input_ids=inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        return tokenizer.decode(
            outputs[0][inputs.shape[1] :], skip_special_tokens=True
        )

# This ensures that when running as a script, we don't execute any async code
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
