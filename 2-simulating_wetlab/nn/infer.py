import os.path

import mlconfig
from tqdm import tqdm
from utils import *
from models.models import get_model
from train_s2s import logits_to_ch


def _main():
    eg_txt()


def eg_txt():
    model_ckpt = './results/ms_nano/Seq2seqRNN'
    input_path = './data/Ours/EncodedStrands.txt'
    output_dir = './results/test_edit_distance_ours_1000_+-10'
    output_path = jpath(output_dir, 'syn.txt')
    min_len = 110  # Minimum output strand length. Set to None if no constraints
    max_len = 130  # This setting is mandatory
    head = 100  # only synthesize for first 'head' strands. Set to None to generate for all.
    device = 'cpu'  # can be cuda

    infer(
        model_ckpt,
        input_path,
        txt_input=True,
        output_path=output_path,
        txt_output=True,
        device=device,
        min_len=min_len,
        max_len=max_len,
        head=head,
    )


def eg_json():
    model_ckpt = './results/ms_nano/Seq2seqRNN'
    input_path = './data/Ours/clean.json'
    output_dir = './results/test_edit_distance_ours_1000_+-10'
    output_path = jpath(output_dir, 'syn.json')
    min_len = 110  # Set to None if no constraints
    max_len = 130  # This setting is mandatory
    head = 100  # only synthesize for first 'head' strands. Set to None to generate for all.
    device = 'cpu'  # can be cuda

    infer(
        model_ckpt,
        input_path,
        txt_input=False,
        output_path=output_path,
        txt_output=False,
        device=device,
        min_len=min_len,
        max_len=max_len,
        head=head,
    )


def infer(model_ckpt_dir, input_path, output_path, device, min_len, max_len,
          head=None, txt_input=False, txt_output=False):
    '''
    Inference with model, allow txt/json input/output.
    '''
    ''' Read input data '''
    if txt_input:
        # Process txt input
        input_dict = read_from_txt(input_path)
    else:
        # Read input from json
        input_dict = read_json(input_path)

    '''Generate noisy strands'''
    ret = infer_from_json(model_ckpt_dir, input_dict, device, min_len, max_len, head)

    '''Save output'''
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if txt_output:
        # Save output to txt
        write_to_txt(ret, output_path)
    else:
        # Save output to json
        save_json(ret, output_path)


def infer_from_json(model_ckpt_dir, input_dict, device, min_len, max_len, head=None):
    '''
    Load the s2s_rnn model,
    initiate with the checkpoint from "model_ckpt_dir",
    generate noisy strands,

    head: only generate for certain clean strands in the front
    '''

    hparam_path = jpath(model_ckpt_dir, 'hparam.yaml')
    hparam = mlconfig.load(hparam_path)
    model_path = jpath(model_ckpt_dir, 'checkpoint.pth')
    net = get_model(hparam['model_name'], hparam).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    tokenizer = S2sTokenizer()
    ret = {}
    net.eval()
    bs = hparam['generate_bs']
    data = input_dict
    data = list(data.items())[:head]

    with torch.no_grad():
        for i in tqdm(range(0, len(data), bs)):
            i_end = i + bs
            clean = [j[1]['ref'] for j in data[i:i_end]]
            enc_inp = tokenizer.batch_tokenize(clean)
            enc_inp = torch.tensor(enc_inp).to(device)  # [B, L]
            enc_out, _ = net.encoder(enc_inp)  # [B, L_in, D]

            syn = []
            for j in range(hparam['generation_coverage']):
                net.decoder.attn.reset()
                c = torch.zeros(
                    enc_out.shape[0], net.decoder.attn_dim, device=enc_out.device
                )
                hs = None
                enc_len = torch.tensor([enc_out.shape[1]]).long()
                input_ids = torch.ones(size=[enc_out.shape[0], 1], dtype=torch.long).to(
                    device)  # [B, 1], <bos> for all seq

                outputs_lst, attn_lst = [], []
                for k in range(max_len + 2 - 1):
                    input_emb = net.decoder.emb(input_ids)  # [B=1, L, D]
                    outputs, hs, c, w = net.decoder.forward_step(
                        input_emb[:, k], hs, c, enc_out, enc_len
                    )
                    logits = net.lm_head(outputs)  # [B, V]
                    outputs_lst.append(outputs)
                    attn_lst.append(w)

                    # greedy sampling
                    ch_id = logits_to_ch(logits).unsqueeze(1).to(device)  # [B,1]

                    input_ids = torch.cat((input_ids, ch_id), dim=1)
                out = input_ids
                syn.append(out)
            syn = torch.stack(syn)  # syn: [coverage, batch, len]
            syn = syn.permute(1, 0, 2)  # [batch, coverage, len]

            for entry, ref, noisy in zip(data[i:i_end], clean, syn):
                id = entry[0]
                noisy = [tokenizer.de_tokenize(i, min_len=min_len) for i in noisy.tolist()]
                ret[id] = {
                    'ref': ref,
                    'syn': noisy,
                }
    return ret


def read_from_txt(input_path):
    with open(input_path) as f:
        data = f.readlines()
    res = {}
    id = 0
    for line in data:
        line = line.strip()
        if len(line) == 0:
            continue
        res[id] = {'ref': line}
        id += 1
    input_dict = res
    return input_dict


def write_to_txt(data, output_path):
    res = []
    for id in data:
        syns = data[id]['syn']
        cluster_str = 'CLUSTER {}\n{}'.format(id, '\n'.join(syns))
        res.append(cluster_str)
    res = '\n'.join(res)
    with open(output_path, 'w') as f:
        f.write(res)


if __name__ == '__main__':
    _main()
