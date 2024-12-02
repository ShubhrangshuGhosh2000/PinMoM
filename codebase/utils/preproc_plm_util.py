import gc
import sys, os
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.insert(0, str(path_root))

import re
import torch
from transformers import T5EncoderModel, T5Tokenizer


# load ProtTrans tl-model for the given type of model
def load_protTrans_model(protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50', device='cpu'):
    # 1. Load necessry libraries including huggingface transformers
    # from transformers import T5EncoderModel, T5Tokenizer
    # import gc

    # 2. Load the vocabulary and ProtTrans Model
    protTrans_model_name_with_path = os.path.join(protTrans_model_path, protTrans_model_name)
    tokenizer = T5Tokenizer.from_pretrained(protTrans_model_name_with_path, do_lower_case=False )
    model = T5EncoderModel.from_pretrained(protTrans_model_name_with_path)
    gc.collect()

    # 3. Load the model into the GPU if avilabile and switch to inference mode
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if(protTrans_model_name == 'prot_t5_xl_half_uniref50-enc'):
        # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
        # model.full() if device=='cpu' else model.half()
        model.half()
    model = model.eval()

    # return the loaded model and tokenizer
    print('returning the loaded model and tokenizer')
    return (model, tokenizer)


def extract_protTrans_plm_feat(prot_seq_lst=None, model=None, tokenizer=None, device='cpu'):
    # print('inside extract_protTrans_plm_feat() method - Start')

    # prot_seq_lst = ["PRTEINO", "SEQWENCE"]
    # 4. convert each protein sequence in a sequence where the amino acids are separated by whitespace and 
    # map rarely occured amino acids (U,Z,O,B) to (X)
    prot_seq_lst_with_space = [" ".join(re.sub(r"[UZOB]", "X", prot_seq)) for prot_seq in prot_seq_lst]

    # 5. Tokenize, encode sequences and load it into the GPU if possibile
    ids = tokenizer.batch_encode_plus(prot_seq_lst_with_space, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # 6. Extracting sequences' features and load it into the CPU if needed
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)
    # embedding = embedding.last_hidden_state.cpu().numpy()
    embedding = embedding.last_hidden_state.cpu()

    plm_1d_embedd_tensor_lst = []
    for seq_num in range(len(embedding)):
        # 7. Remove padding (<pad>) and special tokens (</s>) that is added by ProtT5-XL-UniRef50 model
        seq_len = (attention_mask[seq_num] == 1).sum()
        prot_residue_embed = embedding[seq_num][:seq_len-1]

        # apply pooling column-wise and return the resulting 1d tensor of fixed size (1024) representing per-protein embedding
        plm_1d_embedd_tensor_lst.append(torch.quantile(prot_residue_embed.float(), q=0.5, dim=0))

    return plm_1d_embedd_tensor_lst


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
