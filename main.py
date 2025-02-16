import os
import argparse
import torch
from transformers import LlamaConfig, Trainer, LlamaForCausalLM
import torch.distributed as dist
from utils import LogLoss, SetSeed, GetDataLoader, GetTrainArgs

from modeling_gshard_tutel import GShardTutelModel
from modeling_expert_choice import ExpertChoiceModel
from modeling_dropless import DropLessModel
from modeling_deepseek import DeepSeekModel
from modeling_deepseek_v3 import DeepSeekV3Model
from modeling_ours import OurModel
from modeling_ours_iter import OurModelIter

ModelDict = {
    'Dense': LlamaForCausalLM,
    'GShard':GShardTutelModel, 
    'ExpertChoice':ExpertChoiceModel,
    'DropLess':DropLessModel, 
    'DeepSeek':DeepSeekModel, 
    'DeepSeek-V3':DeepSeekV3Model,
    'Ours': OurModel,
    'Ours-Iter':OurModelIter, 
}
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='Ours-Soft', 
        choices=['Dense', 'GShard', 'ExpertChoice', 'DropLess', 'DeepSeek', 'DeepSeek-V3', 'Ours', 'Ours-Iter',])
    parser.add_argument("--model_config_filepath", type=str, default='configs/LlamaMoE-162M.json',
        choices=['configs/LlamaMoE-162M.json', 'configs/LlamaMoE-317M.json', 'configs/LlamaMoE-599M.json'])
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-rank", type=int, default=-1)

    # TrainingArguments
    parser.add_argument("--batch_size", type=int, default=86)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default='constant_with_warmup')
    parser.add_argument("--warmup_steps", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=20480)

    args = parser.parse_args()

    if args.output_dir == '': 
        args.output_dir = 'logs/{}/'.format(args.model)
        if '317M' in  args.model_config_filepath:
            args.output_dir = args.output_dir[:-1] + '-L/'
        elif '599M' in  args.model_config_filepath:
            args.output_dir = args.output_dir[:-1] + '-XL/'

    world_size, local_rank = int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size, init_method='env://')
    torch.cuda.set_device(local_rank)
    SetSeed(args.seed)

    train_args = GetTrainArgs(args)
    if local_rank == 0: print(train_args)

    config = LlamaConfig.from_pretrained(args.model_config_filepath)
    if args.model == 'Dense':
        config.intermediate_size = config.intermediate_size * 2
    elif 'DeepSeek' in args.model: 
        config.intermediate_size = config.intermediate_size // 4
    model = ModelDict[args.model](config=config).to('cuda:{}'.format(local_rank))
    print('Model {} initialed: {}'.format(args.model, model.device))
    if local_rank == 0: print(config), print(model)
    
    train_dataset = GetDataLoader(args.data_dir)
    print('Dataloader initialed: {}.'.format(local_rank))

    local_log = LogLoss(args.logging_steps * args.gradient_accumulation_steps, local_rank, world_size)
    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs):
            outputs = model(input_ids=inputs['input_ids'], labels=inputs['input_ids'])
            token_loss, gate_loss = outputs.loss, 0
            if args.model != 'Dense' and 'DeepSeek-V3' not in args.model:
                for i in range(len(model.module.model.layers)):
                    gate_loss += model.module.model.layers[i].mlp.l_aux.to(token_loss.device)
                    model.module.model.layers[i].mlp.l_aux = 0
                local_log.update(token_loss, gate_loss)
            return token_loss + gate_loss

    trainer = MyTrainer(model=model, train_dataset=train_dataset, args=train_args)
    trainer.train()
    model.save_pretrained(args.output_dir)
