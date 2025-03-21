from tqdm import tqdm
from scripts.data_transform import format_input
import json
from scripts.ollama_tools import check_if_running, query_model, generate_model_scores

def main():
    model="llama3"
    model="phi-3"
    ollama_running = check_if_running(model)

    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running(model))

    dataset = "alpaca_data_52k.json"
    dataset = "instruction_data_1k.json"
    file_path = f"./dataset_respond/{dataset}-with-response.json"

    with open(file_path, "r") as file:
        test_data = json.load(file)

    # result = query_model("What do Llamas eat?", model)
    # print(result)

    for entry in test_data[:3]:
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
        )
        print("\nDataset response:")
        print(">>", entry['output'])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt))
        print("\n-------------------------")

    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")

if __name__ == "__main__":
    main()