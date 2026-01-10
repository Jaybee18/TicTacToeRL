import os
import glob

from constants import X
from perfect_agent import PerfectAgent
from dqn_agent import DQNAgent
from play import play_against_human
from train import train_agent


def main():
    print("=== TicTacToe Deep Q-Learning ===")
    print("1. Train new agent")
    print("2. Load trained agent and play")
    
    choice = input("Choose option (1 or 2): ")
    
    opponent = PerfectAgent()
    opponent.level = 1
    if choice == "1":
        episodes = int(input("Enter number of training episodes (default 2500): ") or "2500")
        agent = train_agent(agent=DQNAgent(), opponent=opponent, episodes=episodes)
        
        play_more = input("\nDo you want to play against the trained agent? (y/n): ")
        if play_more.lower() == 'y':
            while True:
                play_against_human(agent)
                again = input("\nPlay again? (y/n): ")
                if again.lower() != 'y':
                    break
    elif choice == "2":
        models_dir = "c:\\GitHub\\TicTacToeRL\\models"
        model_files = glob.glob(os.path.join(models_dir, "*.keras"))
        
        if not model_files:
            print(f"No trained models found in {models_dir}")
            print("Please train a model first (option 1 or 2)")
            return
        
        print("\nAvailable models:")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {os.path.basename(model_file)}")
        
        model_choice = int(input(f"\nSelect model (1-{len(model_files)}): ")) - 1
        model_path = model_files[model_choice]
        
        agent = DQNAgent()
        agent.load(model_path)
        agent.epsilon = 0.0  # No exploration when playing
        
        print("\nLoaded trained agent. Ready to play!")
        while True:
            play_against_human(agent)
            again = input("\nPlay again? (y/n): ")
            if again.lower() != 'y':
                break
    else:
        print("Invalid choice. Please run again and select 1, 2, or 3.")


if __name__ == "__main__":
    main()
