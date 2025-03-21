from scripts.model_loader import load_finetune_gpt2_model, load_gpt2_model
from scripts.tools import load_file
from scripts.data_transform import generate, text_to_token_ids, token_ids_to_text
from scripts.data_transform import format_input
import tiktoken
import torch

def main():
    ##################### Load and prepare data #########################
    dataset = "alpaca_data_52k.json"
    dataset = "instruction_data_1k.json"
    file_path = f"./dataset/{dataset}"

    data = load_file(file_path)
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation
    val_data = data[train_portion + test_portion:]
    print("Validation set length:", len(val_data))

    tokenizer = tiktoken.get_encoding("gpt2")
    ######## Load the original gpt2 model ########
    choose_model_gpt2 = "gpt2-medium (355M)"
    model_gpt2, base_config_gpt2 = load_gpt2_model(model_size=choose_model_gpt2, device="cuda" if torch.cuda.is_available() else "cpu")
    ######## Load the instruction fine-tunning gpt2 model ########
    model_name = "gpt2-medium355M-2-instruction_data_1k-sft.pth"
    finetune_model_path = f"./finetune_models/{model_name}"
    model_finetune, base_config_finetune = load_finetune_gpt2_model(model_size=choose_model_gpt2, device="cuda" if torch.cuda.is_available() else "cpu", model_path=finetune_model_path)


    torch.manual_seed(123)
    #### Compare the responses of the two models ####
    for entry in val_data[:2]:
        input_text = format_input(entry)
        
        token_ids_gpt2 = generate(
            model=model_gpt2,
            idx=text_to_token_ids(input_text, tokenizer),
            max_new_tokens=35,
            context_size=base_config_gpt2["context_length"],
            eos_id=50256,
        )
        generated_text_gpt2 = token_ids_to_text(token_ids_gpt2, tokenizer)
        response_text_gpt2 = (
            generated_text_gpt2[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        
        token_ids_finetune = generate(
            model=model_finetune,
            idx=text_to_token_ids(input_text, tokenizer),
            max_new_tokens=35,
            context_size=base_config_finetune["context_length"],
            eos_id=50256,
        )
        generated_text_finetune = token_ids_to_text(token_ids_finetune, tokenizer)
        response_text_finetune = (
            generated_text_finetune[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        
        print(input_text)
        print(f"\nCorrect Response:\n>> {entry['output']}")
        print(f"\nGpt2Model Response:\n>> {response_text_gpt2}")
        print(f"\nFinetuneModel Response:\n>> {response_text_finetune}")
        print("-------------------------------------")


if __name__ == "__main__":
    main()