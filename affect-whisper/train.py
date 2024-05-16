import os, time, argparse
import numpy as np
from tqdm import tqdm
import torch

from src.dataset import WTCAudioDataset, collate
from src.model import AffectWhisper


# CUDA_VISIBLE_DEVICES=1,2 python train.py -e affect --batch_size 16 --num_workers 32 --num_epochs 2 --lr 1e-5 --wd 1e-5 --save /data/rrao/wtc_clinic/models/affect_whisper_v1.pth > exp/aff_10_1e-5_1e-5.txt


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    GPUS = list(range(torch.cuda.device_count()))
    print(f"Number of available GPUs: {GPUS}")
    for i in GPUS:
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
else:
    DEVICE = torch.device('cpu')
    print("CUDA is not available. Only CPU will be used.\n")


def get_arguments():
    parser = argparse.ArgumentParser(description='Training script on `WTC Clinic Audio Files` for Affect-Whisper based on `openai/whisper-small` backbone.')
    parser.add_argument('-e', '--embedding_space', default='affect', choices=['affect', 'roberta'], type=str, help='Specify embedding space to align when fine-tuning')
    parser.add_argument('--batch_size', default=2, type=int, help='Specify a batch_size for training')
    parser.add_argument('--num_workers', default=4, type=int, help='Specify a num_workers for training')
    parser.add_argument('--num_epochs', default=1, type=int, help='Specify a num_epochs for training')
    parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float, help='Specify a learning_rate for training')
    parser.add_argument('--wd', '--weight_decay', default=5e-3, type=float, help='Specify a weight_decay for training')
    parser.add_argument('--shuffle', default=True, choices=[True, False], type=bool, help='Specify whether to shuffle for training')
    parser.add_argument('--save', required=True, type=str, help='Specify a save path for the model checkpoint file')
    parser.add_argument('--load', required=False, type=str, help='Specify a load path to the model checkpoint file to resume training')
    return parser.parse_args()


def train(model, train_dataset, val_dataset, device, batch_size, num_workers, num_epochs, lr, wd, shuffle):
    # Create the DataLoader for the training set
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate)
    # Create the DataLoader for the validation set
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    GRAD_ACC_STEPS = 2

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')

        # TRAIN
        model.train()
        start_time = time.time()
        train_losses = 0.0
        for idx, batch in enumerate(tqdm(train_loader)):
            inputs = batch['inputs'].to(device)
            embeddings = batch['embeddings'].to(device)
            outputs = model(inputs)
            train_loss = torch.square(outputs - embeddings)
            train_loss = torch.mean(torch.mean(train_loss, dim=0))
            loss = train_loss.item()
            print(f'\tBatch: [{idx + 1}]\tTrain Loss: {loss}')
            train_losses += loss

            # Backward and Optimize
            train_loss.backward()
            if (idx + 1) % GRAD_ACC_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Clear GPU Cache
            del inputs, embeddings, outputs
            torch.cuda.empty_cache()

        train_losses /= (idx + 1)    
        hours, remainder = divmod(time.time() - start_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_time = f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'
        print(f'\tAverage Train Loss: {train_losses:.4f}\tTrain Epoch Time: {elapsed_time}')

        # VALIDATION
        model.eval()
        start_time = time.time()
        val_losses = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_loader)):
                inputs = batch['inputs'].to(device)
                embeddings = batch['embeddings'].to(device)
                outputs = model(inputs)
                val_loss = torch.square(outputs - embeddings)
                val_loss = torch.mean(torch.mean(val_loss, dim=0))
                loss = val_loss.item()
                print(f'\tBatch: [{idx + 1}]\tVal Loss: {loss}')
                val_losses += loss

                # Clear GPU Cache
                del inputs, embeddings, outputs
                torch.cuda.empty_cache()
            val_losses /= (idx + 1)
            hours, remainder = divmod(time.time() - start_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed_time = f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'
            print(f'\tAverage Val Loss: {val_losses:.4f}\tVal Epoch Time: {elapsed_time}')


if __name__ == '__main__':
    args = get_arguments()
    emb_space = args.embedding_space
    emb_dims = 2 if emb_space == 'affect' else 1024

    print('Loading WTC Audio Dataset...')
    dataset = WTCAudioDataset(emb_space)
    print('\tDone.\n')

    ds1 = int(0.25 * len(dataset))
    ds2 = len(dataset) - ds1
    dataset, _ = torch.utils.data.random_split(dataset, [ds1, ds2])

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    print('Splitting data to train, val, test...')
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    print('\tDone.\n')

    print('Loading in AffectWhisper Model...')
    model = AffectWhisper(out_dims=emb_dims, device=DEVICE)
    if args.load:
        model_state_dict = torch.load(args.load, map_location=DEVICE)
        if "module." in list(model_state_dict.keys())[0]:
            model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        print('Resuming model training from checkpoint...')
    # DataParallel multi-GPU training (if available)
    if len(GPUS) > 1:
        model = torch.nn.DataParallel(model, device_ids=GPUS)
    print('\tDone.\n')

    print('Starting Training...')
    train(
        model,
        train_dataset,
        val_dataset,
        DEVICE,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        lr=args.lr,
        wd=args.wd,
        shuffle=args.shuffle
    )
    print('Training Complete...')

    print(f'Saving Model to {args.save}')
    model.cpu()
    torch.cuda.empty_cache()
    torch.save(model.state_dict(), args.save)
    print('\tDone.\n')
