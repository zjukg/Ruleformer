''' This module will handle the text generation with beam search. '''
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask
from collections import defaultdict
import numpy, time


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(self, model, opt=None, device=None, base_data=None):

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.max_seq_len = opt.jump + 1
        self.src_pad_idx = opt.src_pad_idx
        self.n_trg_vocab = opt.trg_vocab_size
        self.n_src_vocab = opt.src_vocab_size

        self.device = device

        self.model = model
        self.model.train()
        self.database = [x.to(device) for x in base_data.relations]
        self.filter_dict = base_data.filtered_dict

        self.decode = opt.decode_rule
        if opt.decode_rule:
            self.decode_rule_num = 0
            self.decode_rule_num_filter = 0
            self.the_rel = opt.the_rel
            self.the_rel_min = opt.the_rel_min
            self.the_all = opt.the_all
            self.graph = base_data.neighbors
            self.id2r = base_data.id2r
            self.id2e = base_data.id2e
            self.rules = defaultdict(dict)
            self.decode_file = opt.exps + '/decode_rule.txt'

        self.register_buffer('thr', torch.FloatTensor([1e-20]))


    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = None
        dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)


    def _get_init_state(self, src_seq, src_mask, link=None, length=None):
        enc_output = None
        enc_output, *_ = self.model.encoder(src_seq, src_mask, link=link, length=length)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(1)

        scores = torch.log(best_k_probs).view(self.batch_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx.squeeze()
        return enc_output, gen_seq, scores, dec_output


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step, link=None):
        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(1)
        assert dec_output.size(1)==step

        gen_seq[:, step] = best_k2_idx.squeeze()

        return gen_seq, scores, dec_output


    def forwardAllNLP(self, triples, attention, mode):
        num_pos_rel = self.n_trg_vocab // 2
        batch_size = triples.size(0)
        database = [x.clone().detach() for x in self.database]
        for (h,r,t) in triples:
            if r >= num_pos_rel:
                continue
            idxs = torch.where((h==database[r].indices()[0]) & (t==database[r].indices()[1]))
            database[r].values()[idxs] = 0

        memories = F.one_hot(triples[:,0], num_classes=self.n_src_vocab).float().to_sparse()
        for step in range(self.max_seq_len - 1):
            added_results = torch.zeros(batch_size, self.n_src_vocab).to(self.device)
            for r in range(num_pos_rel):
                for links,atta in zip([database[r], database[r].transpose(0,1)], 
                              [attention[:,step,r], attention[:,step,r+num_pos_rel]]):
                    added_results = added_results + torch.sparse.mm(memories, links).to_dense() * atta.unsqueeze(1)
            added_results = added_results + memories.to_dense() * attention[:, step, -1].unsqueeze(1)
            added_results = added_results / torch.max(self.thr, torch.sum(added_results, dim=1).unsqueeze(1))
            memories = added_results.to_sparse()
        memories = memories.to_dense()
        targets = F.one_hot(triples[:,2], num_classes=self.n_src_vocab).float()
        final_loss = - torch.sum(targets * torch.log(torch.max(self.thr, memories)), dim=1)
        batch_loss = torch.mean(final_loss)
        if mode != 'train':
            for i in range(batch_size):
                for idx in self.filter_dict[(triples[i,0].item(), triples[i,1].item())]:
                    if idx != triples[i,2].item():
                        memories[i, idx] = 0
        idxs = torch.argsort(memories, descending=True)
        indexs = torch.where(idxs==triples[:,2].unsqueeze(-1))[1].tolist()
        return batch_loss, indexs


    def forward(self, src_seq, trg=None, tailIndexs=None, link=None, padding=300, mode='train', length=None):
        batch_size = trg.size(0)
        self.batch_size = batch_size
        self.init_seq = trg[:,1].unsqueeze(-1).clone().detach()
        self.blank_seqs = trg[:,1].unsqueeze(-1).repeat(1,self.max_seq_len).clone().detach()
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        enc_output, gen_seq, scores, dec_output = self._get_init_state(src_seq, src_mask, link=link, length=length)
        for step in range(2, self.max_seq_len):
            dec_output = self._model_decode(gen_seq[:, :step].clone().detach(), enc_output, src_mask)
            gen_seq, scores, dec_output = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step, link=link)

        if self.decode:
            self.decode_rule(dec_output, trg)
            loss, index = 0, [0]
        else:
            loss, index = self.forwardAllNLP(trg, dec_output, mode)

        return gen_seq, loss, index


    def decode_rule(self, dec_output, trg):
        relation_attention_list=dec_output
        batch_size = trg.size(0)
        num_step = self.max_seq_len - 1
        for batch in range(batch_size):
            paths = {t+1: [] for t in range(num_step)}
            # paths at hop 0, in the format of ([rel1,..],[ent1,..],weight)
            paths[0] = [([-1], [trg[batch, 0].item()], 1.)]
            relation_attentions = relation_attention_list[batch]
            for step in range(num_step):
                if not paths[step]:
                    break
                relation_attention_ori = relation_attentions[step]
                for rels, pths, wei in paths[step]:
                    if pths[-1] not in self.graph:
                        continue
                    # select relations(including self-loop) connected to the tail of each path
                    sel = torch.LongTensor(list(self.graph[pths[-1]].keys()) + [self.n_trg_vocab-1])
                    relation_attention = torch.zeros(self.n_trg_vocab).to(self.device)
                    relation_attention[sel] = relation_attention_ori[sel].clone()
                    rel_att_max = torch.max(relation_attention).item()
                    relation_attention /= rel_att_max

                    for rr in torch.nonzero(relation_attention > max(self.the_rel, self.the_rel_min/rel_att_max)): # relations which exceed threshold
                        rr = rr.item()
                        if rr == self.n_trg_vocab - 1: # <slf>
                            paths[step+1].append((rels+[rr], pths+[pths[-1]], wei*relation_attention[rr].item()))
                        elif rr in self.graph[pths[-1]].keys():
                            for tail in self.graph[pths[-1]][rr]:
                                paths[step+1].append((rels+[rr], pths+[tail], wei*relation_attention[rr].item()))

            for path in paths[step+1]:
                rels, pths, wei = path
                if path[2] > self.the_all:
                    self.decode_rule_num += 1
                    print('\rWrite {}-{} Rule(s)'.format(self.decode_rule_num, self.decode_rule_num_filter), end='')
                    head_rule = self.id2r[trg[batch, 1].item()]
                    rule_body = '^'.join([self.id2r[r] for r in rels[1:]])
                    try:
                        self.rules[head_rule][rule_body].append(wei)
                    except KeyError:
                        self.rules[head_rule][rule_body] = [wei]
                        self.decode_rule_num_filter += 1
                        with open(self.decode_file ,'a') as f:
                            f.write(head_rule+'<-'+rule_body+'\n')
