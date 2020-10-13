import json
import logging
import os
import random
import sys

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.tokenization_openai import OpenAIGPTTokenizer

from config import get_trainer_config
from config import InputConfig
from model.dataset import FacebookDataset
from model.gpt2_model import GPT2DoubleHeadsModel
from model.gpt2_model import GPT2EncoderDecoderModel
from model.openai_model import OpenAIGPTEncoderDecoderModel
from model.openai_model import OpenAIGPTLMHeadModel
from model.trainer import Trainer
from model.utils import config_logger
from model.utils import f1_score
from model.utils import open
from model.utils import set_seed
from model.seq2seq import TransformerSeq2Seq
from model.seq2seq_vocab import Seq2seqVocab
from new_metrics import nlp_metrics


class DummyWriter:
    """ Used for distributed training (from NVIDIA apex example).
        A dummy logger used so that only the main process write and log informations.
    """
    def __init__(self, *input, **kwargs):
        self.log_dir = "runs/dummy_logs/"

    def add_scalar(self, *input, **kwargs):
        pass

def modify_tokenizer(tokenizer, data_type):
    additional_special_tokens = ['<info_bos>', '<info_eos>', '<talker1_bos>', '<talker1_eos>', '<talker2_bos>',
                                 '<talker2_eos>']
    if data_type == 'emoji':
        with open('datasets/emoji_talk/emojis.json', 'r') as f:
            emojis = json.load(f)['emojis']
        additional_special_tokens.extend(emojis)
    if data_type == 'daily':
        with open('datasets/DailyDialog/daily.json', 'r') as f:
            topic_tokens = json.load(f)
        additional_special_tokens.extend(topic_tokens)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>',
                                  'additional_special_tokens': additional_special_tokens})
    tokenizer.eos_id, tokenizer.bos_id, tokenizer.pad_id = tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id
    tokenizer.sent_dialog_id = tokenizer.bos_token_id
    tokenizer.info_dialog_id, tokenizer.info_bos_id = tokenizer.added_tokens_encoder['<info_bos>'], \
                                                      tokenizer.added_tokens_encoder[
                                                          '<info_bos>']
    tokenizer.info_eos_id = tokenizer.added_tokens_encoder['<info_eos>']
    tokenizer.talker1_dialog_id, tokenizer.talker1_bos_id = tokenizer.added_tokens_encoder['<talker1_bos>'], \
                                                            tokenizer.added_tokens_encoder['<talker1_bos>']
    tokenizer.talker1_eos_id = tokenizer.added_tokens_encoder['<talker1_eos>']
    tokenizer.talker2_dialog_id, tokenizer.talker2_bos_id = tokenizer.added_tokens_encoder['<talker2_bos>'], \
                                                            tokenizer.added_tokens_encoder['<talker2_bos>']
    tokenizer.talker2_eos_id = tokenizer.added_tokens_encoder['<talker2_eos>']
    return tokenizer, len(additional_special_tokens) + 3

def main():
    args = InputConfig().args

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.ERROR)
    logger = logging.getLogger(__file__)
    if args.server_ip and args.server_port and args.local_rank in [-1, 0]:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    trainer_config = get_trainer_config(args)
    # with open('/apdcephfs/share_916081/rainyucao/transformer_chatbot_experiments/test_log', 'w') as f:
    #     a = []
    #     a.append('args local rank is ' + str(args.local_rank) + '\n')
    #     a.append('cuda count' + str(torch.cuda.device_count()) + '\n')
    #     if args.local_rank not in [-1, 0] and torch.cuda.device_count() == 1:
    #         args.local_rank = -1
    #     a.append('args local rank is ' + str(args.local_rank) + '\n')
    #     f.writelines(a)

    # Log only on main process
    if args.local_rank not in [-1, 0]:
        sys.stdout = open("./runs/log_distributed_{}".format(args.local_rank), "w")  # dump sdtout
        writer = DummyWriter()
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.ERROR)
        logger = logging.getLogger(__file__)
    else:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        if args.single_input:
            comment = '_{}_{}_single'.format(args.model_type, args.data_type)
        else:
            if args.model_type == 'seq2seq':
                comment = '_seq2seq_multi_{}_{}'.format(args.data_type, args.attention_fusion_type)
            else:
                comment = '_{}_{}_{}_{}_{}'.format(args.model_type, args.data_type, args.attention_fusion_type,
                           ('sm' if args.shared_module == 1 else 'nm'), ('sa' if args.shared_attention == 1 else 'na'))
        logdir = os.path.join('runs', current_time + comment)
        writer = SummaryWriter(logdir=logdir)
        logger = config_logger(os.path.join(logdir, 'train.log'))

    log_dir = writer.logdir
    logger.info("Training args: {}".format(args))
    logger.info("trainer config: {}".format(trainer_config))
    interrupt_checkpoint_path = os.path.join(log_dir, trainer_config.interrupt_checkpoint_path)
    last_checkpoint_path = os.path.join(log_dir, trainer_config.last_checkpoint_path)
    best_checkpoint_path = os.path.join(log_dir, 'best_model')
    logger.info("Logging to {}".format(log_dir))  # Let's save everything on an experiment in the ./runs/XXX/directory
    if args.local_rank in [-1, 0]:
        with open(os.path.join(log_dir, "trainer_config.json"), "w") as f:
            json.dump(trainer_config, f)

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)

    parsed_train_data, parsed_valid_data, parsed_test_data = None, None, None
    if args.model_type == 'gpt':
        if args.single_input:
            model = OpenAIGPTLMHeadModel.from_pretrained('./openai-gpt')
        else:
            model = OpenAIGPTEncoderDecoderModel.from_pretrained('./openai-gpt')
        tokenizer = OpenAIGPTTokenizer.from_pretrained('./openai-gpt')
    elif args.model_type == 'dialogpt':
        if args.single_input:
            model = GPT2DoubleHeadsModel.from_pretrained('./dialogpt')
        else:
            model = GPT2EncoderDecoderModel.from_pretrained('./dialogpt')
        tokenizer = GPT2Tokenizer.from_pretrained('./dialogpt')
    elif args.model_type == 'seq2seq':
        seq2seq_vocab = Seq2seqVocab(trainer_config.train_datasets, trainer_config.valid_datasets,
                                 trainer_config.test_datasets, args.vocab_path, data_type=args.data_type)
        tokenizer = seq2seq_vocab.vocab
        parsed_train_data, parsed_valid_data, parsed_test_data = seq2seq_vocab.all_data[0], seq2seq_vocab.all_data[1], \
                                                                 seq2seq_vocab.all_data[2]
        args.dialog_embeddings = False
        model = TransformerSeq2Seq(args.emb_dim, args.hidden_dim, args.num_layers, args.heads, args.depth_size,
                               args.filter_size, tokenizer, args.pretrained_emb_file, args.pointer_gen, logger,
                                multi_input=not args.single_input, attention_fusion_type=args.attention_fusion_type)
    else:
        if args.single_input:
            model = GPT2DoubleHeadsModel.from_pretrained('./gpt2-small')
        else:
            model = GPT2EncoderDecoderModel.from_pretrained('./gpt2-small')
        tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-small')


    if args.model_type in ['gpt', 'dialogpt', 'gpt2']:
        tokenizer, additional_length = modify_tokenizer(tokenizer, args.data_type)
        model.embeddings_size = 768
        model.n_embeddings = len(tokenizer)
        model.shared_attention = (args.shared_attention == 1)
        model.shared_module = (args.shared_module == 1)
        model.attention_fusion_type = args.attention_fusion_type
        model.single_input = args.single_input
        if args.model_type == 'gpt':
            model_embedding_weight = model.transformer.tokens_embed.weight
            model.transformer.tokens_embed = nn.Embedding(model.n_embeddings, 768)
            model.lm_head = nn.Linear(768, model.n_embeddings, bias=False)
            model.transformer.tokens_embed.weight.data[:-additional_length, :] = model_embedding_weight.data
            model.transformer.tokens_embed.weight.data[-additional_length:, :] = 0
            model.lm_head.weight = model.transformer.tokens_embed.weight
        else:
            model_embedding_weight = model.transformer.wte.weight
            model.transformer.wte = nn.Embedding(model.n_embeddings, 768)
            model.lm_head = nn.Linear(768, model.n_embeddings, bias=False)
            model.transformer.wte.weight.data[:-additional_length, :] = model_embedding_weight.data
            model.transformer.wte.weight.data[-additional_length:, :] = 0
            model.lm_head.weight = model.transformer.wte.weight

        if not args.single_input:
            model.reload_module_dict()
        model.sent_dialog_id = tokenizer.sent_dialog_id
    model.talker1_id = tokenizer.talker1_bos_id
    model.talker2_id = tokenizer.talker2_bos_id

    model.padding_idx = tokenizer.pad_id
    model.n_pos_embeddings = 512

    model.bos_id = tokenizer.bos_id
    model.eos_id = tokenizer.eos_id
    model.beam_size = args.beam_size
    model.diversity_groups = 1
    model.max_seq_len = 32
    model.dialog_embeddings = args.dialog_embeddings
    model.bs_temperature = args.bs_temperature
    model.bs_nucleus_p = args.bs_nucleus_p
    model.annealing_topk = args.annealing_topk
    model.length_penalty_coef = args.length_penalty
    model.vocab = None
    model.annealing = args.annealing
    model.diversity_coef = args.diversity_coef
    model.sample = False
    model.inference_mode = args.inference_mode
    model.response_k = args.response_k

    logger.info('loading datasets')
    train_dataset = FacebookDataset(trainer_config.train_datasets, tokenizer,
                                    max_lengths=model.n_pos_embeddings - 1,  # A bit restrictive here
                                    dialog_embeddings=args.dialog_embeddings,
                                    cache=trainer_config.train_datasets_cache,
                                    use_start_end=False,
                                    negative_samples=trainer_config.negative_samples,
                                    augment=trainer_config.persona_augment,
                                    aug_syn_proba=trainer_config.persona_aug_syn_proba,
                                    limit_size=trainer_config.limit_train_size,
                                    max_history_size=trainer_config.max_history_size,
                                    single_input=args.single_input,
                                    data_type=args.data_type,
                                    parsed_data=parsed_train_data)
    valid_dataset = FacebookDataset(trainer_config.valid_datasets, tokenizer,
                                    max_lengths=model.n_pos_embeddings - 1,  # A bit restrictive here
                                    dialog_embeddings=args.dialog_embeddings,
                                    cache=trainer_config.valid_datasets_cache,
                                    use_start_end=False,
                                    negative_samples=-1,  # Keep all negative samples
                                    augment=False,
                                    aug_syn_proba=0.0,
                                    limit_size=trainer_config.limit_eval_size,
                                    max_history_size=trainer_config.max_history_size,
                                    single_input=args.single_input,
                                    data_type=args.data_type,
                                    parsed_data=parsed_valid_data)
    test_dataset = FacebookDataset(trainer_config.test_datasets, tokenizer,
                                   max_lengths=model.n_pos_embeddings - 1,  # A bit restrictive here
                                   dialog_embeddings=args.dialog_embeddings,
                                   cache=trainer_config.test_datasets_cache,
                                   use_start_end=False,
                                   negative_samples=-1,  # Keep all negative samples
                                   augment=False,
                                   aug_syn_proba=0.0,
                                   limit_size=trainer_config.limit_eval_size,
                                   max_history_size=trainer_config.max_history_size,
                                   single_input=args.single_input,
                                   data_type=args.data_type,
                                   parsed_data=parsed_test_data)
    logger.info('train dataset {} valid dataset {} test dataset {}'
                .format(len(train_dataset), len(valid_dataset), len(test_dataset)))

    # if args.local_rank != -1:
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '12355'
    #
    #     # initialize the process group
    #     torch.distributed.init_process_group("nccl", rank=args.local_rank, world_size=1)
    #     n = torch.cuda.device_count()
    #     device_ids = list(range(args.local_rank * n, (args.local_rank + 1) * n))
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device('cuda', args.local_rank)
    #     transformer.distribute(device_ids[0], device_ids)
    '''Normal training will use normal trainer'''
    model_trainer = Trainer(model,
                            train_dataset,
                            writer,
                            logger=logger,
                            valid_dataset=valid_dataset,
                            test_dataset=test_dataset,
                            train_batch_size=trainer_config.train_batch_size,
                            batch_split=trainer_config.batch_split,
                            test_batch_size=trainer_config.test_batch_size,
                            lr=trainer_config.lr,
                            lr_warmup=trainer_config.lr_warmup,
                            weight_decay=trainer_config.weight_decay,
                            s2s_weight=trainer_config.s2s_weight,
                            lm_weight=trainer_config.lm_weight,
                            risk_weight=trainer_config.risk_weight,
                            hits_weight=trainer_config.hits_weight,
                            single_input=trainer_config.single_input,
                            n_jobs=trainer_config.n_jobs,
                            clip_grad=trainer_config.clip_grad,
                            device=device,
                            ignore_idxs=tokenizer.all_special_ids,
                            local_rank=args.local_rank,
                            apex_level=None,
                            apex_loss_scale=trainer_config.apex_loss_scale,
                            linear_schedule=trainer_config.linear_schedule,
                            n_epochs=trainer_config.n_epochs,
                            evaluate_full_sequences=trainer_config.evaluate_full_sequences,
                            full_input=trainer_config.full_input,
                            uncertainty_loss=args.uncertainty_loss,
                            best_model_path=best_checkpoint_path,
                            extra_module_lr_rate=args.extra_module_lr_rate,
                            no_persona=args.no_persona)

    if args.load_last:
        state_dict = torch.load(trainer_config.load_last, map_location=device)
        model_trainer.load_state_dict(state_dict)

    # helpers -----------------------------------------------------
    def external_metrics_func(full_references, full_predictions, epoch, metric=None, is_best=False):
        if epoch == -1:
            if is_best:
                references_file_path = os.path.join(writer.logdir, 'test_references_file')
                predictions_file_path = os.path.join(writer.logdir, 'test_predictions_file_best')
            else:
                references_file_path = os.path.join(writer.logdir, 'test_references_file')
                predictions_file_path = os.path.join(writer.logdir, 'test_predictions_file_last')
        else:
            references_file_path = os.path.join(writer.logdir, trainer_config.eval_references_file)
            predictions_file_path = os.path.join(writer.logdir, trainer_config.eval_predictions_file + "_{}".format(epoch))
        if not os.path.exists(references_file_path):
            with open(references_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(full_references))
        # print(len(full_predictions))
        with open(os.path.join(writer.logdir, 'tt.json'), 'w') as f:
            json.dump(full_predictions, f)
        with open(predictions_file_path, 'w', encoding='utf-8') as f:
            if len(full_predictions[-1]) == 0:
                full_predictions[-1] = 'a '
            f.write('\n'.join(full_predictions))

        bleu, bleu_list, nist, nist_list, nist_bleu, nist_bleu_list, s_dist, c_dist, entropy, meteor, \
                rouge_l, f1_score, avg_length = nlp_metrics(references_file_path, predictions_file_path, root_path=log_dir)

        metrics = {'meteor': meteor, 'avg_len': avg_length, 'rouge-l': rouge_l, 'bleu': bleu, 'nist': nist,
                   'nist-bleu': nist_bleu, 'f1': f1_score}
        for name, metric in (('bleu', bleu_list), ('nist', nist_list), ('nist_bleu', nist_bleu_list), ('entropy', entropy),
                             ('sentence_div', s_dist), ('corpus_div', c_dist)):
            for i, m in enumerate(metric, 1):
                metrics['{}_{}'.format(name, i)] = m

        return metrics

    def save_func(epoch):
        if epoch != -1:
            torch.save(model_trainer.model.state_dict(), last_checkpoint_path)
            logger.info('Model on Epoch %d has been saved', epoch)

    def sample_text_func(epoch):
        n_samples = 0
        model_trainer.model.eval()
        samples_idxs = random.sample(range(len(valid_dataset)), n_samples)
        samples = [valid_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target, _ in samples:
            contexts = [torch.tensor([c], dtype=torch.long, device=model_trainer.device) for c in [persona_info, dialog] if len(c) > 0]
            prediction = model_trainer.model.predict(contexts)[0]

            persona_info_str = tokenizer.ids2string(persona_info[1:-1])
            dialog_str = tokenizer.ids2string(dialog)
            dialog_str = dialog_str.replace(tokenizer.talker1_bos, '\n\t- ').replace(tokenizer.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(tokenizer.talker1_eos, '').replace(tokenizer.talker2_eos, '')
            target_str = tokenizer.ids2string(target[1:-1])
            prediction_str = tokenizer.ids2string(prediction)

            logger.info('\n')
            logger.info('Persona info:\n\t{}'.format(persona_info_str))
            logger.info('Dialog:{}'.format(dialog_str))
            logger.info('Target:\n\t{}'.format(target_str))
            logger.info('Prediction:\n\t{}'.format(prediction_str))

    def test_func(epoch):
        if (epoch+1) % trainer_config.test_period == 0:
            metric_funcs = {'f1_score': f1_score}
            model_trainer.test(metric_funcs, external_metrics_func, epoch)

    def f1_risk(predictions, targets):
        scores = f1_score(predictions, targets, average=False)
        assert all([0 <= s <= 1.0 for s in scores])
        return [1 - s for s in scores]

    def get_risk_metric_func(risk_metric):
        """ risk_metric selected in:
            f1, meteor, avg_len, nist_{1, 2, 3, 4}, entropy_{1, 2, 3, 4}, div_{1, 2}, bleu_{1, 2, 3, 4}
        """
        def external_metric_risk(predictions, targets):
            string_targets = list(tokenizer.ids2string(t) for t in targets)
            string_predictions = list(tokenizer.ids2string(t) for t in predictions)
            metrics = [external_metrics_func([t], [p], epoch=-1, metric=risk_metric) for p, t in zip(string_predictions, string_targets)]

            if any([s in risk_metric for s in ['entropy', 'nist', 'avg_len']]):
                return [-m for m in metrics]

            assert all([0 <= s <= 1.0 for s in metrics]), metrics

            return [1 - m for m in metrics]

        if risk_metric == 'f1':
            return f1_risk

        return external_metric_risk

    # helpers -----------------------------------------------------

    try:
        model_trainer.train(after_epoch_funcs=[save_func, sample_text_func, test_func],
                            risk_func=get_risk_metric_func(trainer_config.risk_metric))
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        if args.local_rank in [-1, 0]:
            torch.save(model_trainer.state_dict(), interrupt_checkpoint_path)
        raise e


if __name__ == '__main__':
    main()
