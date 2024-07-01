import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import BertTokenizer
from accelerate import Accelerator

import yaml
import time
from loguru import logger
from tqdm.auto import tqdm
from argparse import ArgumentParser

from utils import seed_environment, check_config, dataloader_init
from model import model_init, Transformer

def eval(dataloader: DataLoader):
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(config['checkpoint_zh'])

    epoch_loss = 0
    num_steps = len(dataloader)
    progress_bar = tqdm(range(num_steps), desc="🌟 Eval", unit="step")
    for batch in dataloader:
        batch_zh, batch_en = batch
        loss = model.teacher_forcing(batch_en, batch_zh)

        texts = batch_zh['input_ids']
        greedy_results = model.greedy_decode(batch_en)
        beam_results = model.beam_search_decode(batch_en)
        result = tokenizer.decode(texts[0], skip_special_tokens=False)
        greedy_result = tokenizer.decode(greedy_results[0], skip_special_tokens=False)
        beam_result = tokenizer.decode(beam_results[0], skip_special_tokens=False)
        print(f'\ntruth:\n{result}\ngreedy:\n{greedy_result}\nbeam:\n{beam_result}\n')

        epoch_loss += loss.item()
        progress_bar.update(1)

    epoch_loss /= len(dataloader)
    return epoch_loss

if __name__ == '__main__':
    """ setup with config """
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    seed_environment(seed=config['seed'])
    accelerator = Accelerator()
    config['device'] = accelerator.device

    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default=config['name'])
    parser.add_argument('--num_shards', type=int, default=config['num_shards'])
    parser.add_argument('--save_log', action='store_true', help='default false')
    parser.add_argument('--save_model', action='store_true', help='default false')
    args = parser.parse_args()

    config['name'] = args.name
    config['save_log'] = args.save_log
    config['save_model'] = args.save_model
    config['num_shards'] = args.num_shards
    check_config(config)

    if config['save_log'] == True:
        local_time = time.localtime(time.time())
        date_time = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)
        log_path = os.path.join(config['log_path'], "{}_shards_{}/{}".format(config['name'], config['num_shards'], date_time))
        writer = SummaryWriter(log_dir=log_path)

    """ initialize model, dataloader and optimizer """
    train_dataloader, valid_dataloader, test_dataloader = dataloader_init(config_init=config)
    model: Transformer = model_init(config_init=config)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    logger.info('👏 Everything is ready!')


    """ start training """
    num_train_epochs = config['num_train_epochs']
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    progress_bar = tqdm(range(num_training_steps), desc="💫 Train", unit="step")
    global_step = 0
    for epoch in range(1, num_train_epochs+1):
        model.train()
        epoch_loss = 0
        for batch in train_dataloader:
            batch_zh, batch_en = batch
            loss = model.teacher_forcing(batch_en, batch_zh)
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
            logger.info(f'(train) step: {global_step}, loss: {loss.item()}')
            if config['save_log']:
                writer.add_scalar('train/loss', loss, global_step)
            epoch_loss += loss.item()
            # epoch_loss1 += loss1.item()
            # epoch_loss2 += loss2.item()
            global_step += 1
            progress_bar.update(1)
        epoch_loss /= num_update_steps_per_epoch
        logger.info(f'(train) epoch: {epoch}, epoch_loss: {epoch_loss}')

        epoch_valid_loss = eval(valid_dataloader)
        # print(result)
        logger.info(f'(valid) epoch: {epoch}, epoch_loss: {epoch_valid_loss}')
        if config['save_log']:
            writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
            writer.add_scalar('valid/epoch_loss', epoch_valid_loss, epoch)

        if accelerator.is_main_process and config['save_model'] and epoch % config['num_epochs_save_model'] == 0:
            accelerator.wait_for_everyone()
            logger.info("🌏 Everyone is here!")

            save_path = os.path.join(
                config['output_path'], 
                "{}_shards_{}".format(config['name'], config['num_shards']), 
                "checkpoint-{}".format(global_step)
            )
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            accelerator.save_state(save_path)
            logger.info("Saved state to {}".format(save_path), main_process_only=True)
