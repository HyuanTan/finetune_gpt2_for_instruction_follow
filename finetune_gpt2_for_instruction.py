from scripts.tools import load_file, custom_collate_fn
from scripts.data_transform import format_input, generate, text_to_token_ids, token_ids_to_text
import tiktoken
from functools import partial
import torch
from torch.utils.data import DataLoader
from scripts.instruction_dataset import InstructionDataset
from scripts.model_loader import load_small_test_model, load_gpt2_model
from scripts.train_tools import calc_loss_loader, train_model_simple
import time
from scripts.tools import plot_losses
import re
import json
from tqdm import tqdm

def main(test_mode=False):
    ##################### Load and prepare dataset #########################
    num_epochs = 2
    num_workers = 0
    batch_size = 2 #4 # 8
    
    dataset = "alpaca_data_52k.json"
    # dataset = "instruction_data_1k.json"
    file_path = f"./dataset/{dataset}"
    data = load_file(file_path)
    data = data[:2000]
    print("Number of entries:", len(data))
    ''' 
    print("Example entry:\n", data[50])
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_response)
    '''
    
    train_portion = int(len(data) * 0.80)  # 80% for training
    test_portion = int(len(data) * 0.15)    # 15% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    # Use very small subset for testing purposes
    if test_mode:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]
    print("\nTraining set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    ## Batching the data
    tokenizer = tiktoken.get_encoding("gpt2")
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    torch.manual_seed(123)
    #########################
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    # print("Train loader:")
    # for inputs, targets in train_loader:
    #     print(inputs.shape, targets.shape)
    
    
    ####### Load pretrained model ########
    if test_mode:
        CHOOSE_MODEL = "Small test model"
        model, BASE_CONFIG = load_small_test_model()
    else:
        CHOOSE_MODEL = "gpt2-medium (355M)"
        model, BASE_CONFIG = load_gpt2_model(model_size=CHOOSE_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    
    ####### Finetuning the model ########
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    
    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen, best_model_state_dict= train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    
     #### Saving results ########
    print("Generating responses")
    if test_mode:
        test_data = test_data[:10]
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    test_data_path = f"./dataset_respond/{dataset}-with-response.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)
    print(f"Responses saved as {test_data_path}")
    
     
    #### Save the model ########
    file_name = f"./finetune_models/{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-{num_epochs}-{dataset}-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

    file_name = f"./finetune_models/{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-{num_epochs}-{dataset}-best_sft.pth"
    torch.save(best_model_state_dict, file_name)
    print(f"Best model (lowest val loss) saved as {file_name}")


if __name__ == "__main__":
    enable_test_mode = False
    # enable_test_mode = True
    main(test_mode=enable_test_mode)