import torch, numpy, pickle, random, time, argparse
from collections import defaultdict


class DataBase():
    def __init__(self, data_path, rel_pad=0, subgraph='') -> None:
        with open(f'{data_path}/entities.txt') as e, open(f'{data_path}/relations.txt') as r:
            self.ents = ['<pad>'] + [x.strip() for x in e.readlines()]
            self.rels = [x.strip() for x in r.readlines()]
            self.pos_rels = len(self.rels)
            self.rels += ['inv_'+x for x in self.rels] + ['<slf>']
            self.e2id = {self.ents[i]:i for i in range(len(self.ents))}
            self.r2id = {self.rels[i]:i for i in range(len(self.rels))}
            self.id2r = {i:self.rels[i] for i in range(len(self.rels))}
            self.id2e = {i:self.ents[i] for i in range(len(self.ents))}
        self.data = {}
        with open(f'{data_path}/train.txt') as f:
            train = [item.strip().split('\t') for item in f.readlines()]
            self.data['train'] = list({(self.e2id[h],self.r2id[r],self.e2id[t]) for h,r,t in train})
        with open(f'{data_path}/test.txt') as f:
            test = [item.strip().split('\t') for item in f.readlines()]
            self.data['test'] = list({(self.e2id[h],self.r2id[r],self.e2id[t]) for h,r,t in test})
        with open(f'{data_path}/valid.txt') as f:
            valid = [item.strip().split('\t') for item in f.readlines()]
            self.data['valid'] = list({(self.e2id[h],self.r2id[r],self.e2id[t]) for h,r,t in valid})

        indices = [[] for _ in range(self.pos_rels)]
        values = [[] for _ in range(self.pos_rels)]
        for h,r,t in self.data['train']:
            indices[r].append((h,t))
            values[r].append(1)
        indices = [torch.LongTensor(x).T for x in indices]
        values = [torch.FloatTensor(x) for x in values]
        size = torch.Size([len(self.ents),len(self.ents)])
        self.relations = [torch.sparse.FloatTensor(indices[i], values[i], size).coalesce() for i in range(self.pos_rels)]

        self.filtered_dict = defaultdict(set)
        triplets = self.data['train'] + self.data['valid'] + self.data['test']
        for triplet in triplets:
            self.filtered_dict[(triplet[0], triplet[1])].add(triplet[2])
            self.filtered_dict[(triplet[2], triplet[1]+self.pos_rels)].add(triplet[0])

        self.neighbors = defaultdict(dict)
        for h, r, t in self.data['train']:
            try:
                self.neighbors[h][r].add(t)
            except KeyError:
                self.neighbors[h][r] = set([t])
            try:
                self.neighbors[t][r+self.pos_rels].add(h)
            except KeyError:
                self.neighbors[t][r+self.pos_rels] = set([h])
        for h in self.neighbors:
            self.neighbors[h] = {r:list(ts) for r,ts in self.neighbors[h].items()}

        self.nebor_relation = torch.ones(len(self.e2id), len(self.r2id))
        for h, r, t in self.data['train']:
            self.nebor_relation[h][r] += 1
            self.nebor_relation[t][r+self.pos_rels] += 1
        for e in self.e2id.values():
            if e not in self.neighbors.keys():
                self.nebor_relation[e][2*self.pos_rels] += 1
        self.nebor_relation = torch.log(self.nebor_relation)
        self.nebor_relation /= self.nebor_relation.sum(1).unsqueeze(1)

        if subgraph:
            with open(subgraph, 'rb') as db:
                self.subgraph = pickle.load(db)

    def getinfo(self):
        return len(self.ents), len(self.rels)

    def extract_without_token(self, head, JUMP, MAXN, PADDING):
        subgraph = [head]
        relation = []
        length = [0]
        for jump in range(JUMP):
            length.append(len(subgraph))
            for parent in range(length[jump], length[jump+1]):
                for r in self.neighbors[subgraph[parent]]:
                    if len(self.neighbors[subgraph[parent]][r]) > MAXN:
                        print(f'J{subgraph[parent]}-{r}-{len(self.neighbors[subgraph[parent]][r])}', end=' ')
                        # continue
                        random.shuffle(self.neighbors[subgraph[parent]][r])
                    for t in self.neighbors[subgraph[parent]][r][:MAXN]:
                        try:
                            pos = subgraph.index(t)
                            relation.append((parent, pos, r))
                        except ValueError:
                            subgraph.append(t)
                            pos = subgraph.index(t)
                            relation.append((parent, pos, r))
        length.append(len(subgraph))
        if length[-1] > PADDING or not relation: # subgraph is too big
            subgraph = subgraph[:PADDING]
            length[-1] = len(subgraph)
        RELA = set()
        for i,j,r in relation:
            if i<PADDING and j<PADDING:
                RELA.add((i,j,r))
                inv_r = r + self.pos_rels * ((r<self.pos_rels) * 2 - 1)
                RELA.add((j,i,inv_r))

        return subgraph, numpy.array(list(RELA)).T, length


class pickleDataset(torch.utils.data.Dataset):
    def __init__(self, database, opt, mode='train'):
        super().__init__()
        self.triples = database.data[mode]
        self.neighbors = database.neighbors
        self.nrels = len(database.rels)
        self.pos_rels = self.nrels // 2
        self.padding = opt.padding
        self.subgraph = database.subgraph

    def __getitem__(self, index):
        H,R,T = self.triples[index]
        sub1, rela1, trg1, tails1, leng1 = self.getsubgraph(H, R, T)
        sub2, rela2, trg2, tails2, leng2 = self.getsubgraph(T, R+self.pos_rels, H)
        return sub1, rela1, trg1, tails1, leng1, sub2, rela2, trg2, tails2, leng2

    def getsubgraph(self, H, R, T):
        subgraph,relations,length = self.subgraph[H]
        assert(subgraph.index(H)==0)
        rela_mat = torch.zeros(self.padding, self.padding, self.nrels)
        rela_mat[relations] = 1

        try:
            t = subgraph.index(T)
            inv_r = R + self.pos_rels * ((R<self.pos_rels) * 2 - 1)
            rela_mat[subgraph.index(H),t,R] = 0
            rela_mat[t,subgraph.index(H),inv_r] = 0
        except ValueError:
            pass

        subgraph += [0 for _ in range(self.padding - len(subgraph))]
        return torch.LongTensor(subgraph), torch.FloatTensor(rela_mat), torch.tensor([H,R,T]), torch.LongTensor([0]), torch.LongTensor(length)

    def __len__(self) -> int:
        return len(self.triples)

    @staticmethod
    def collate_fn(data):
        subs =  torch.stack([d[0] for d in data] + [d[5] for d in data], dim=0)
        relas = torch.stack([d[1] for d in data] + [d[6] for d in data], dim=0)
        trgs =  torch.stack([d[2] for d in data] + [d[7] for d in data], dim=0)
        tails = torch.stack([d[3] for d in data] + [d[8] for d in data], dim=0)
        lengs = torch.stack([d[4] for d in data] + [d[9] for d in data], dim=0)
        return subs, relas, trgs, tails, lengs


def main(jump, base_data, path, maxn, padding):
    subgraph = dict()
    cnt = []
    false = []
    for head in range(1, len(base_data.ents)): # 0 for <pad>
        res = base_data.extract_without_token(head, jump, maxn, padding) # head, JUMP, MAXN, PADDING
        try:
            print(f'{head}\tlen:{res[2]}')
            subgraph[head] = res
            cnt.append(res[2][-1])
        except Exception:
            false.append(base_data.ents[head])
    print(len(false), false)
    with open(f'{path}/subgraph{jump}', 'wb') as db:
        pickle.dump(subgraph, db)
        print(sum(cnt) / len(cnt))
    lis = {i:0 for i in range(padding+1)}
    for item in cnt:
        lis[item] += 1
    print({k:v for k,v in lis.items() if v})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-data', type=str, required=True)
    parser.add_argument('-maxN', type=int, required=True)
    parser.add_argument('-jump', type=int, required=True)
    parser.add_argument('-padding', type=int, required=True)
    opt = parser.parse_args()
    # wn18rr MAXN=10
    # umls   MAXN=40
    # 237    MAXN=70
    # 237    MaxN=40
    path = 'DATASET/' + opt.data
    base_data = DataBase(path)
    # main(1, base_data, path, 100, 40)
    main(opt.jump, base_data, path, opt.maxN, opt.padding)
    # main(3, base_data, path, 10, 100)
    # main(3, base_data)
