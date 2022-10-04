''' Translate input text with trained model. '''

import torch
import argparse, os, time

from transformer.Models import Transformer
from transformer.Translator import Translator
from transformer.dataset import DataBase, pickleDataset
from transformer.Optim import ScheduledOptim
import torch.optim as optim
import numpy as np
import random


def load_model(opt, device, nebor_relation):
    model = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        opt.src_pad_idx,
        -1,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj,
        n_position = opt.padding,
        data = opt.data,
        opt = opt,
        nebor_relation = nebor_relation.to(device)).to(device)
    return model

def hit_mrr(hits, starttime):
    return 'MRR:{:.5f} @1:{:.5f} @3:{:.5f} @10:{:.5f} LOS:{:.5f} Time:{:.1f}secs'.format(hits[10]/hits[12], hits[0]/hits[12], hits[0:3].sum()/hits[12],hits[0:10].sum()/hits[12],hits[11]/hits[12], time.time()-starttime)

def run(translator, data_loader, id2r, mode, optimizer, device, padding, epoch, logfile, starttime, decode):
    pred_line = []
    hits = np.zeros(13) # [0:10] for hit, [10] for mrr, [11] for loss, [12] for cnt
    for i,(subgraph , link, target, tailIndexs, length) in enumerate(data_loader):
        pred_seq, loss, indexL = translator(subgraph.to(device), target.to(device), tailIndexs.to(device), link.to(device), padding, mode, length.to(device))
        if decode == True:
            continue
        print(f'\r {mode} {epoch}-{i}/{len(data_loader)}', end='    ')
        for j, index in enumerate(indexL):
            if index < 10:
                hits[index] += 1
            hits[10] += 1/(index+1)
            hits[12] += 1
        hits[11] += loss.item()
        if mode=='train':
            loss.backward()
            optimizer.step_and_update_lr()
            optimizer.zero_grad()
        print(hit_mrr(hits, starttime), end='       ')
    print(f'\r         {mode}-{epoch}  ' + hit_mrr(hits, starttime) + '     ')
    with open(logfile, 'a') as log:
        log.write(f'{mode}-{epoch}  ' + hit_mrr(hits, starttime) + '\n')
    return pred_line


def main(tofetch):
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-data', type=str, required=True)
    parser.add_argument('-jump', type=int, default=0, required=True)
    parser.add_argument('-padding', type=int, default=0, required=True)
    parser.add_argument('-ckpt', type=str, default='')
    parser.add_argument('-desc', type=str, required=True)
    parser.add_argument('-exps', type=str, default='EXPS/')
    parser.add_argument('-subgraph', type=str, default='')
    parser.add_argument('-savestep', type=int)

    parser.add_argument('-epoch', type=int, default=10)

    # Transformers parameter
    parser.add_argument('-d_v', type=int, default=50)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=2)

    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=400)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=31)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-label_smoothing', action='store_true')

    parser.add_argument('-output', default='pred.txt', help="""Path to output the predictions (each line will be the decoded sequence""")
    parser.add_argument('-batch_size', type=int, default=5)

    # get rule
    parser.add_argument('-decode_rule', default=False, action="store_true", help='whether decode the rule')
    parser.add_argument('-the_rel', default=0.6, type=float, help='relative threshold of the next relation')
    parser.add_argument('-the_rel_min', default=0.3, type=float, help='absolute threshold of the next relation')
    parser.add_argument('-the_all', default=0.1, type=float, help='absolute threshold of the whole rule')

    opt = parser.parse_args()
    opt.d_k = opt.d_v
    opt.d_inner_hid = opt.d_model = opt.d_k * opt.n_head
    opt.desc += f'-j{opt.jump}'
    opt.desc += time.strftime("_%Y%m%d_%H:%M:%S", time.localtime())
    opt.exps = os.path.join(opt.exps, opt.desc)
    opt.subgraph = opt.data+f'/subgraph{opt.jump}' if opt.subgraph=='' else opt.data+f'/subgraph{opt.subgraph}'
    if opt.decode_rule:
        assert(opt.ckpt)
        opt.epoch = 1
        opt.savestep = 0
    print(f'save at:{opt.exps}')
    os.mkdir(opt.exps)
    with open(opt.exps + '/options.txt', 'w') as option:
        for k,v in sorted(opt.__dict__.items(), key=lambda x: x[0]):
            option.write(f'{k} = {v}\n')
    logfile = opt.exps + '/log.txt'

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
    opt.d_word_vec = opt.d_model

    base_data = DataBase(opt.data, subgraph=opt.subgraph)
    opt.src_vocab_size, opt.trg_vocab_size = base_data.getinfo()
    train_data = pickleDataset(base_data, opt, mode='train')
    valid_data = pickleDataset(base_data, opt, mode='valid')
    test_data = pickleDataset(base_data, opt, mode='test')
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=pickleDataset.collate_fn) # , num_workers=16
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=opt.batch_size, shuffle=False, collate_fn=pickleDataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=pickleDataset.collate_fn)

    opt.src_pad_idx = base_data.e2id['<pad>']

    device = torch.device('cuda')

    translator = Translator(
        model = load_model(opt, device, base_data.nebor_relation),
        opt = opt,
        device = device,
        base_data = base_data).to(device)

    if opt.ckpt:
        translator.load_state_dict(torch.load(opt.ckpt))
    
    tofetch['decode'] = opt.decode_rule
    if opt.decode_rule:
        tofetch['rules'] = translator.rules
        tofetch['dir'] = opt.exps

    optimizer = ScheduledOptim(
        optim.Adam(translator.parameters(), betas=(0.9, 0.98), eps=1e-09),
                   opt.lr_mul, opt.d_model, opt.n_warmup_steps)
    starttime = time.time()
    for epoch in range(opt.epoch):
        if not opt.decode_rule:
            run(translator, train_loader, base_data.id2r, 'train', optimizer, device, opt.padding, epoch+1, logfile, starttime, opt.decode_rule)
        if opt.savestep and (epoch + 1) % opt.savestep == 0:
            torch.save(translator.state_dict(), f'{opt.exps}/Translator{epoch+1}.ckpt')
        if opt.decode_rule or (epoch + 1) % 5 == 0:
            with torch.no_grad():
                if opt.decode_rule:
                    run(translator, train_loader, base_data.id2r, 'train', optimizer, device, opt.padding, epoch+1, logfile, starttime, opt.decode_rule)
                run(translator, valid_loader, base_data.id2r, 'valid', optimizer, device, opt.padding, epoch+1, logfile, starttime, opt.decode_rule)
                run(translator, test_loader, base_data.id2r, 'test', optimizer, device, opt.padding, epoch+1, logfile, starttime, opt.decode_rule)

    print('[Info] Finished.')


if __name__ == "__main__":
    try:
        tofetch = {}
        main(tofetch)
    except KeyboardInterrupt:
        print('\nExited')
    if tofetch['decode']:
        print('Writing Rules...')
        with open(tofetch['dir']+'/rules.txt' ,'w') as rr:
            for head_rel in tofetch['rules']:
                for rel_body in tofetch['rules'][head_rel]:
                    wei = tofetch['rules'][head_rel][rel_body]
                    rr.write(f'{len(wei)}-{sum(wei)/len(wei)} {head_rel} <- {rel_body}\n')