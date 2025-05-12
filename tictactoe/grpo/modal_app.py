import os
import modal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = modal.App("grpo-tik-tak-toe")

# Get Weights & Biases and HuggingFace API keys from environment
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Create image with all required dependencies
image = (
    modal.Image.debian_slim()
    # Install procps package for pkill command needed by art library
    .apt_install("procps")
    .pip_install(
        "numpy==1.26.4",
        "openai>=1.74.0",
        "openpipe>=4.50.0",
        "openpipe-art>=0.3.5",
        "pydantic>=2.11.4",
        "python-dotenv>=1.1.0",
        "torch>=2.5.1",
        "unsloth", 
        "wandb",   
        "tqdm",
        "huggingface_hub>=0.20.3"     
    )
    # Add environment variables for API keys BEFORE adding local files
    .env({
        "WANDB_API_KEY": WANDB_API_KEY,
        "HF_TOKEN": HF_TOKEN
    })
    # Now add local files as the last step
    .add_local_dir(local_path=".", remote_path="/root/code")
)

# Persistent volume to store models and data
volume = modal.Volume.from_name("grpo-finetune")

REPO_ROOT = "/root"
DATA_PATH = "/data"
MODEL_PATH = f"{DATA_PATH}/models"  # Path to save models in the volume

@app.function(
    image=image, 
    gpu="H100", 
    timeout=60 * 60,  # 1 hour timeout
    volumes={DATA_PATH: volume},
)
async def train(
    base_model="Qwen/Qwen2.5-3B-Instruct", # "banachspace/Qwen-0.5B-TTT-SFT-merged", # "Qwen/Qwen2.5-3B-Instruct",
    hf_username="banachspace",
    push_to_hub=True,
    epochs=1,
    num_rollouts=1
):
    """
    Train the tic-tac-toe model and push it to HuggingFace Hub
    
    Args:
        hf_repo_id: Repository ID for HuggingFace Hub (default: banachspace/Qwen-2.5-0.5B-Instruct-TicTacToe-GRPO)
        push_to_hub: Whether to push to HuggingFace Hub (default: True)
        epochs: Number of training epochs (default: 10)
        num_rollouts: Number of rollouts per epoch (default: 20)
    
    Returns:
        dict: Information about the trained model
    """
    import sys
    import os

    model_name = base_model.split("/")[1]
    hf_repo_id = f"{hf_username}/{model_name}-TicTacToe-GRPO"
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Set art storage path to our volume
    os.environ["ART_STORAGE_PATH"] = MODEL_PATH
    
    # Add code directory to Python path
    sys.path.append("/root/code")
    
    # Import and run the main function
    from tic_tac_toe import main
    result = await main(
        base_model=base_model, 
        hf_repo_id=hf_repo_id,
        push_to_hub=push_to_hub,
        epochs=epochs,
        num_rollouts=num_rollouts,
        run_demo=True  # Skip demo game to avoid errors
    )
    
    print(f"Training complete!")
    
    if result.get("huggingface_repo"):
        print(f"Model pushed to HuggingFace Hub: {result['huggingface_repo']['repo_url']}")
        print(f"You can find it at: https://huggingface.co/{hf_repo_id}")
    elif result.get("huggingface_error"):
        print(f"Error pushing to HuggingFace Hub: {result['huggingface_error']}")
    
    return result

@app.function(
    image=image,
    gpu="T4",  # Using smaller GPU for inference
    volumes={DATA_PATH: volume},
)
async def run_inference(model_path=None, hf_repo_id=None):
    """
    Run inference with the trained model.
    If model_path is None, will use the latest model in the volume.
    If hf_repo_id is provided, will load the model from HuggingFace Hub.
    
    Args:
        model_path: Path to the model in the volume (optional)
        hf_repo_id: HuggingFace Hub repository ID to use (optional)
    """
    import sys
    import os
    import glob
    
    # Add code directory to Python path
    sys.path.append("/root/code")
    
    # If HF repo ID is provided, use that model
    if hf_repo_id:
        print(f"Using model from HuggingFace Hub: {hf_repo_id}")
        # Import components needed for inference
        from tic_tac_toe import generate_game, get_completion
        from tic_tac_toe import render_board, apply_agent_move, check_winner, get_opponent_move
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model from HuggingFace Hub
        try:
            model = AutoModelForCausalLM.from_pretrained(hf_repo_id)
            tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
            print(f"Successfully loaded model from HuggingFace Hub: {hf_repo_id}")
            
            # Play a demonstration game
            play_game_with_model(model, tokenizer)
            return {"source": "huggingface", "repo_id": hf_repo_id}
            
        except Exception as e:
            print(f"Error loading model from HuggingFace Hub: {e}")
            return {"error": str(e)}
    
    # First, show all available model paths in the volume for debugging
    print(f"Searching for models in: {MODEL_PATH}")
    # Check the volume contents
    os.system(f"find {DATA_PATH} -type d | sort")
    
    # List all potential model directories
    model_pattern = f"{MODEL_PATH}/**/**/????/"
    all_model_dirs = glob.glob(model_pattern, recursive=True)
    print(f"\nFound {len(all_model_dirs)} potential model directories:")
    for d in all_model_dirs:
        print(f"  - {d}")
    
    # Try a different pattern if needed
    art_pattern = f"{MODEL_PATH}/.art/**/models/**/????/"
    art_model_dirs = glob.glob(art_pattern, recursive=True)
    print(f"\nFound {len(art_model_dirs)} models in .art directory:")
    for d in art_model_dirs:
        print(f"  - {d}")
    
    # Try to find models outside .art directory
    root_pattern = f"{DATA_PATH}/**/models/**/????/"
    root_model_dirs = glob.glob(root_pattern, recursive=True)
    print(f"\nFound {len(root_model_dirs)} models at root level:")
    for d in root_model_dirs:
        print(f"  - {d}")
    
    # If no specific model path is provided, try to use the latest model
    if model_path is None:
        # Try different patterns to find model directories
        patterns = [
            f"{MODEL_PATH}/**/**/????/",
            f"{MODEL_PATH}/.art/**/models/**/????/",
            f"{DATA_PATH}/**/models/**/????/"
        ]
        
        for pattern in patterns:
            model_dirs = sorted(glob.glob(pattern, recursive=True))
            if model_dirs:
                model_path = model_dirs[-1]  # Get the latest model
                print(f"Using latest model found: {model_path}")
                break
        
        if not model_path:
            print("No models found. Please specify a model path or run training first.")
            return {"error": "No models found in volume."}
    
    print(f"Using model from: {model_path}")
    
    # Check if the model files actually exist
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        return {"error": f"Model path {model_path} does not exist."}
    
    print(f"Model files in {model_path}:")
    os.system(f"ls -la {model_path}")
    
    # Import components needed for inference
    from tic_tac_toe import generate_game, get_completion
    from tic_tac_toe import render_board, apply_agent_move, check_winner, get_opponent_move
    import torch
    from unsloth import FastLanguageModel
    
    try:
        # Load the model
        peft_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=16384,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(peft_model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"error": f"Failed to load model: {str(e)}"}
    
    # Play a demonstration game
    game = generate_game()
    move_number = 0
    
    messages = [
        {
            "role": "system",
            "content": f"You are a tic-tac-toe player. You are playing against an opponent. Always choose the move most likely to lead to an eventual win. Return your move as an XML object with a single property 'move', like so: <move>A1</move>. Optional moves are 'A1', 'B3', 'C2', etc. You are the {game['agent_symbol']} symbol.",
        },
    ]
    
    print(f"Starting a game with the agent as {game['agent_symbol']}")
    
    # If agent is 'o', opponent (x) goes first
    if game["agent_symbol"] == "o":
        starting_opponent_move = get_opponent_move(game)
        game["board"][starting_opponent_move[0]][starting_opponent_move[1]] = game["opponent_symbol"]
        print(f"Opponent went first with position {chr(65+starting_opponent_move[0])}{starting_opponent_move[1]+1}")
    
    # Main game loop
    while check_winner(game["board"]) is None:
        rendered_board = render_board(game)
        messages.append({"role": "user", "content": rendered_board})
        
        print(f"\nBoard state (move {move_number}):")
        print(rendered_board)
        
        # Generate agent's move
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to("cuda")
        
        try:
            content = get_completion(peft_model, tokenizer, inputs)
            messages.append({"role": "assistant", "content": content})
            
            print(f"Agent's move: {content}")
            
            # Apply the agent's move to the board
            apply_agent_move(game, content)
            move_number += 1
            
            print(f"Updated board:")
            print(render_board(game))
            
            # Check if the game is over after agent's move
            winner = check_winner(game["board"])
            if winner is not None:
                break
                
            # Opponent's turn
            opponent_move = get_opponent_move(game)
            game["board"][opponent_move[0]][opponent_move[1]] = game["opponent_symbol"]
            move_number += 1
            
            print(f"Opponent's move: {chr(65+opponent_move[0])}{opponent_move[1]+1}")
            
        except Exception as e:
            print(f"Error with agent's move: {e}")
            return {"error": f"Game aborted due to error: {str(e)}"}
    
    # Game is over, print results
    winner = check_winner(game["board"])
    final_board = render_board(game)
    
    print(f"\nGame finished in {move_number} moves")
    print(f"Final board:\n{final_board}")
    
    if winner == game["agent_symbol"]:
        result = "Agent won! 💪"
    elif winner == game["opponent_symbol"]:
        result = "Agent lost! 😢"
    elif winner == "draw":
        result = "Game ended in a draw! 🤷‍♂️"
    
    print(result)
    return {"result": result, "moves": move_number, "final_board": final_board}

def play_game_with_model(model, tokenizer):
    """Helper function to play a game with a model loaded from HuggingFace Hub"""
    # Import necessary modules
    from tic_tac_toe import generate_game, render_board, apply_agent_move, check_winner, get_opponent_move
    import torch
    
    # Create a new game
    game = generate_game()
    move_number = 0
    
    # Set up system message
    messages = [
        {
            "role": "system",
            "content": f"You are a tic-tac-toe player. You are playing against an opponent. Always choose the move most likely to lead to an eventual win. Return your move as an XML object with a single property 'move', like so: <move>A1</move>. Optional moves are 'A1', 'B3', 'C2', etc. You are the {game['agent_symbol']} symbol.",
        },
    ]
    
    print(f"Starting a game with the agent as {game['agent_symbol']}")
    
    # If agent is 'o', opponent (x) goes first
    if game["agent_symbol"] == "o":
        starting_opponent_move = get_opponent_move(game)
        game["board"][starting_opponent_move[0]][starting_opponent_move[1]] = game["opponent_symbol"]
        print(f"Opponent went first with position {chr(65+starting_opponent_move[0])}{starting_opponent_move[1]+1}")
    
    # Main game loop
    while check_winner(game["board"]) is None:
        rendered_board = render_board(game)
        messages.append({"role": "user", "content": rendered_board})
        
        print(f"\nBoard state (move {move_number}):")
        print(rendered_board)
        
        # Generate agent's move
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            content = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": content})
            
            print(f"Agent's move: {content}")
            
            # Apply the agent's move to the board
            apply_agent_move(game, content)
            move_number += 1
            
            print(f"Updated board:")
            print(render_board(game))
            
            # Check if the game is over after agent's move
            winner = check_winner(game["board"])
            if winner is not None:
                break
                
            # Opponent's turn
            opponent_move = get_opponent_move(game)
            game["board"][opponent_move[0]][opponent_move[1]] = game["opponent_symbol"]
            move_number += 1
            
            print(f"Opponent's move: {chr(65+opponent_move[0])}{opponent_move[1]+1}")
            
        except Exception as e:
            print(f"Error with agent's move: {e}")
            return {"error": f"Game aborted due to error: {str(e)}"}
    
    # Game is over, print results
    winner = check_winner(game["board"])
    final_board = render_board(game)
    
    print(f"\nGame finished in {move_number} moves")
    print(f"Final board:\n{final_board}")
    
    if winner == game["agent_symbol"]:
        result = "Agent won! 💪"
    elif winner == game["opponent_symbol"]:
        result = "Agent lost! 😢"
    elif winner == "draw":
        result = "Game ended in a draw! 🤷‍♂️"
    
    print(result)
    return {"result": result, "moves": move_number, "final_board": final_board}

@app.function(
    image=image, 
    gpu="T4",  # Using a smaller GPU for quick testing
    timeout=30 * 60,  # 30 minutes timeout
    volumes={DATA_PATH: volume},
)
async def train_small(hf_repo_id="banachspace/Qwen-0.5B-TicTacToe-GRPO-Test", push_to_hub=True):
    """
    Train a small model for quick testing with fewer resources
    
    Args:
        hf_repo_id: Repository ID for HuggingFace Hub
        push_to_hub: Whether to push to HuggingFace Hub (default: True)
    
    Returns:
        dict: Information about the trained model
    """
    import sys
    import os
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Set art storage path to our volume
    os.environ["ART_STORAGE_PATH"] = MODEL_PATH
    
    # Add code directory to Python path
    sys.path.append("/root/code")
    
    # Import and run the main function with the small model
    from tic_tac_toe import main
    result = await main(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",  # Smaller model
        hf_repo_id=hf_repo_id,
        push_to_hub=push_to_hub,
        epochs=2,  # Just 2 epochs for quick testing
        num_rollouts=5,  # Fewer rollouts
        run_demo=False  # Skip demo game to avoid errors
    )
    
    print(f"Training complete!")
    
    if result.get("huggingface_repo"):
        print(f"Model pushed to HuggingFace Hub: {result['huggingface_repo']['repo_url']}")
        print(f"You can find it at: https://huggingface.co/{hf_repo_id}")
    elif result.get("huggingface_error"):
        print(f"Error pushing to HuggingFace Hub: {result['huggingface_error']}")
    
    return result

@app.local_entrypoint()
def main():
    print("Running main entrypoint...")
    print()
    print("Available commands:")
    print("1. Train full model (H100, ~1 hour):")
    print("   modal run tictactoe/grpo/modal_app.py::train")
    print()
    print("2. Train small test model (T4, ~15 minutes):")
    print("   modal run tictactoe/grpo/modal_app.py::train_small")
    print()
    print("3. Run inference with a HF model:")
    print("   modal run tictactoe/grpo/modal_app.py::run_inference --hf-repo-id=YOUR_USERNAME/MODEL_NAME")
    print()
    print("Additional options:")
    print("- Customize epochs: modal run tictactoe/grpo/modal_app.py::train --epochs=5")
    print("- Customize rollouts: modal run tictactoe/grpo/modal_app.py::train --num-rollouts=10")

if __name__ == "__main__":
    # When run as a script, deploy the app
    app.deploy()