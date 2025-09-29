import torch, math, json
import torch.nn.functional as F
import pandas as pd
import numpy as np

CODES = {"A": 0, "T": 3, "G": 1, "C": 2, 'N': 4}
COMPL = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}

def n2id(n: str) -> int:
    return CODES[n.upper()]

def revcomp(seq: str) -> str:
    return ''.join(COMPL[n.upper()] for n in reversed(seq))

class Seq2Tensor:
    def __call__(self, seq: str) -> torch.Tensor:
        indices = [n2id(base) for base in seq]
        arr = torch.tensor(indices, dtype=torch.long)
        one_hot = F.one_hot(arr, num_classes=5)[:, :4].float()  # Use only A, C, G, T
        return one_hot.T  # shape: (4, seq_len)

def pad_sequence(seq, seqsize: int) -> str:
    total_pad = seqsize - len(seq)
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return 'N' * left_pad + seq + 'N' * right_pad

def file_length(file: str) -> int:
    with open(file,'r') as f:
        lines=f.readlines()
        return len(lines)

def preprocess_data(df: pd.DataFrame, 
                    seqsize: int, 
                    species: str,
                    plasmid_path: str | None) -> pd.DataFrame:
    if species == 'human':
        return preprocess_human_data(df, seqsize=seqsize)
    elif species == 'yeast':
        assert plasmid_path is not None
        return preprocess_yeast_data(df, seqsize=seqsize, plasmid_path=plasmid_path)
    else:
        raise Exception("Dataset not implemented")

def preprocess_human_data(df: pd.DataFrame, seqsize: int) -> pd.DataFrame:
    df=df.copy()
    df['seq'] = df['seq'].apply(pad_sequence,args=(seqsize,))
    return df

def preprocess_yeast_data(df: pd.DataFrame, 
                    seqsize: int,
                    plasmid_path: str) -> pd.DataFrame:
    '''
    Pads training sequences on the 5-end with nucleotides from the plasmid
    '''
    left_adapter = "TGCATTTTTTTCACATC"
    if df.iloc[0,0][:len(left_adapter)]!=left_adapter:
        return df
    
    with open(plasmid_path) as json_file:
        plasmid = json.load(json_file)

    df = df.copy()
    INSERT_START = plasmid.find('N' * 80)
    add_part = plasmid[INSERT_START-seqsize:INSERT_START]
    df.seq = df.seq.apply(lambda x:  add_part + x[len(left_adapter):])
    df.seq = df.seq.str.slice(-seqsize, None)
    return df

def add_revcomp(df: pd.DataFrame,
                revcomp_same_batch: bool=False,
                batch_size: int=4096) -> pd.DataFrame:
    df = df.copy()
    rev = df.copy()
    rev['seq'] = df['seq'].apply(revcomp)
    rev['rev'] = 1
    df['rev'] = 0
    if revcomp_same_batch:
        df_list = []
        half_batch=batch_size//2
        df_size=len(df)
        for batch_num in range(math.ceil(df_size/half_batch)):
            df_list.append(pd.concat([df[batch_num*half_batch:min((batch_num+1)*half_batch, df_size)], 
                                      rev[batch_num*half_batch:min((batch_num+1)*half_batch, df_size)]])
                                      .reset_index(drop=True))
        return pd.concat(df_list,ignore_index=True)
    else:
        return pd.concat([df, rev],ignore_index=True)

def add_singleton_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_singleton"] = np.array([x.is_integer() for x in df['expr']])
    return df 

def preprocess_tsv(path: str, 
                   seqsize: int, 
                   species: str,
                   plasmid_path: str | None,
                   revcomp_same_batch: bool=False,
                   batch_size: int=1024) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = ['seq', 'expr']
    df = preprocess_data(df, seqsize=seqsize, species=species, plasmid_path=plasmid_path)
    df = add_revcomp(df,revcomp_same_batch=revcomp_same_batch,batch_size=batch_size)
    if species == 'yeast':
        df = add_singleton_column(df)
    return df

class SeqExprDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 seqsize: int,
                 use_single_channel: bool=False,
                 use_reverse_channel: bool=True):
        self.df = df.reset_index(drop=True)
        self.encoder = Seq2Tensor()
        self.seqsize = seqsize
        self.use_single_channel = use_single_channel
        self.use_reverse_channel = use_reverse_channel

    def __len__(self):
        return len(self.df)
    
    def transform(self, x: str) -> torch.Tensor:
        return self.encoder(x)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        seq = self.transform(row['seq']) # type: ignore
        to_concat = [seq]
        
        if self.use_reverse_channel:
            rev = torch.full( (1, self.seqsize), row['rev'], dtype=torch.float32) # type: ignore
            to_concat.append(rev)
            
        if self.use_single_channel:
            single = torch.full( (1, self.seqsize) , row['is_singleton'], dtype=torch.float32) # type: ignore
            to_concat.append(single)

        if len(to_concat) > 1:    
            X = torch.concat(to_concat, dim=0)
        else:
            X = seq
            
        y = torch.tensor(row['expr'], dtype=torch.float32)
        return {"x": X.float(), "y": y}
    
class DataloaderWrapper:
    def __init__(self, dataloader: torch.utils.data.DataLoader, batch_per_epoch: int):
        self.dataloader = dataloader
        self.batch_per_epoch = batch_per_epoch
        self.iterator = iter(dataloader)

    def __iter__(self):
        for _ in range(self.batch_per_epoch):
            try:
                yield next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)

    def __len__(self):
        return self.batch_per_epoch

def prepare_dataloader(
    tsv_path: str,
    seqsize: int,
    species: str,
    plasmid_path: str = 'data/yeast/plasmid.json',
    batch_size: int = 1024,
    num_workers: int = 4,
    shuffle: bool = True,
    generator: torch.Generator = None,
    batch_per_epoch: int = None,
    revcomp_same_batch: bool = False
) -> torch.utils.data.DataLoader:
    
    df = preprocess_tsv(path=tsv_path, 
                        seqsize=seqsize,
                        species=species,
                        plasmid_path=plasmid_path,
                        revcomp_same_batch=revcomp_same_batch,
                        batch_size=batch_size)
    use_single_channel = species == 'yeast'
    dataset = SeqExprDataset(df=df, 
                             seqsize=seqsize,
                             use_single_channel=use_single_channel)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator
    )

    if batch_per_epoch is not None:
        return DataloaderWrapper(dataloader, batch_per_epoch)
    return dataloader