"""
Make new predictions with a pre-trained model. One of --seqs or --embeddings is required.
"""
from __future__ import annotations
import argparse
import datetime
import sys


import numpy as np
import pandas as pd
import torch
from scipy.special import comb
from tqdm import tqdm
from typing import Callable, NamedTuple, Optional


from ..alphabets import Uniprot21
from ..fasta import parse
from ..language_model import lm_embed, lm_embed_pd
from ..utils import log, load_hdf5_parallel


class PredictionArguments(NamedTuple):
    cmd: str
    device: int
    embeddings: Optional[str]
    outfile: Optional[str]
    seqs: str
    model: str
    thresh: Optional[float]
    func: Callable[[PredictionArguments], None]


def add_args(parser):
    """
    Create parser for command line utility

    :meta private:
    """

    parser.add_argument(
        "--pairs", help="Candidate protein pairs to predict", required=True
    )
    parser.add_argument("--model", help="Pretrained Model", required=True)
    parser.add_argument("--seqs", help="Protein sequences in .fasta format")
    parser.add_argument("--embeddings", help="h5 file with embedded sequences")
    parser.add_argument("-o", "--outfile", help="File for predictions")
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.5,
        help="Positive prediction threshold - used to store contact maps and predictions in a separate file. [default: 0.5]",
    )
    return parser


def main(args):
    """
    Run new prediction from arguments.

    :meta private:
    """
    if args.seqs is None and args.embeddings is None:
        log("One of --seqs or --embeddings is required.")
        sys.exit(0)

    csvPath = args.pairs
    modelPath = args.model
    outPath = args.outfile
    seqPath = args.seqs
    embPath = args.embeddings
    device = args.device
    threshold = args.thresh

    # Set Outpath
    if outPath is None:
        outPath = datetime.datetime.now().strftime(
            "%Y-%m-%d-%H:%M.predictions"
        )

    logFilePath = outPath + ".log"
    logFile = open(logFilePath, "w+")

    # Set Device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        log(
            f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}",
            file=logFile,
            print_also=True,
        )
    else:
        log("Using CPU", file=logFile, print_also=True)

    # Load Model
    try:
        log(f"Loading model from {modelPath}", file=logFile, print_also=True)
        if use_cuda:
            model = torch.load(modelPath).cuda()
            model.use_cuda = True
        else:
            model = torch.load(
                modelPath, map_location=torch.device("cpu")
            ).cpu()
            model.use_cuda = False
    except FileNotFoundError:
        log(f"Model {modelPath} not found", file=logFile, print_also=True)
        logFile.close()
        sys.exit(1)

    # Load Pairs
    try:
        log(f"Loading pairs from {csvPath}", file=logFile, print_also=True)
        pairs = pd.read_csv(csvPath, sep="\t", header=None)
        all_prots = set(pairs.iloc[:, 0]).union(set(pairs.iloc[:, 1]))
    except FileNotFoundError:
        log(f"Pairs File {csvPath} not found", file=logFile, print_also=True)
        logFile.close()
        sys.exit(1)

    # ############################## commented code -start ##############################
    # # Load Sequences or Embeddings
    # if embPath is None:
    #     try:
    #         names, seqs = parse(seqPath, "r")
    #         seqDict = {n: s for n, s in zip(names, seqs)}
    #     except FileNotFoundError:
    #         log(
    #             f"Sequence File {seqPath} not found",
    #             file=logFile,
    #             print_also=True,
    #         )
    #         logFile.close()
    #         sys.exit(1)
    #     log("Generating Embeddings...", file=logFile, print_also=True)
    #     embeddings = {}
    #     for n in tqdm(all_prots):
    #         embeddings[n] = lm_embed(seqDict[n], use_cuda)
    # else:
    #     log("Loading Embeddings...", file=logFile, print_also=True)
    #     embeddings = load_hdf5_parallel(embPath, all_prots)
    # ############################## commented code -end ##############################


    # Make Predictions
    log("Making Predictions...", file=logFile, print_also=True)

    # ############################## added code -start ##############################
    try:
        names, seqs = parse(seqPath, "r")
        seqDict = {n: s for n, s in zip(names, seqs)}
    except FileNotFoundError:
        log(
            f"Sequence File {seqPath} not found",
            file=logFile,
            print_also=True,
        )
        logFile.close()
        sys.exit(1)
    # ############################## added code -end ##############################

    n = 0
    outPathAll = f"{outPath}.tsv"
    outPathPos = f"{outPath}.positive.tsv"
    # cmap_file = h5py.File(f"{outPath}.cmaps.h5", "w")
    model.eval()

    with open(outPathAll, "w+") as f:
        with open(outPathPos, "w+") as pos_f:
            with torch.no_grad():
                for _, (n0, n1) in tqdm(
                    pairs.iloc[:, :2].iterrows(), total=len(pairs)
                ):
                    n0 = str(n0)
                    n1 = str(n1)
                    if n % 50 == 0:
                        f.flush()
                    n += 1
                    # ############################## added code -start ##############################
                    # p0 = embeddings[n0]
                    # p1 = embeddings[n1]

                    p0 = lm_embed(seqDict[n0], use_cuda)
                    p1 = lm_embed(seqDict[n1], use_cuda)
                    # ############################## added code -end ##############################
                    if use_cuda:
                        p0 = p0.cuda()
                        p1 = p1.cuda()
                    try:
                        cm, p = model.map_predict(p0, p1)
                        p = p.item()
                        f.write(f"{n0}\t{n1}\t{p}\n")
                        if p >= threshold:
                            pos_f.write(f"{n0}\t{n1}\t{p}\n")
                        #     cm_np = cm.squeeze().cpu().numpy()
                        #     dset = cmap_file.require_dataset(
                        #         f"{n0}x{n1}", cm_np.shape, np.float32
                        #     )
                        #     dset[:] = cm_np
                    except RuntimeError as e:
                        log(
                            f"{n0} x {n1} skipped ({e})",
                            file=logFile,
                        )

    logFile.close()
    # cmap_file.close()


def main_pd(pairs=None, model=None, pre_trn_mdl=None, alphabet=None, seqs=None, threshold=0.5):
    """
    Run new prediction from arguments.

    """
    # csvPath = args.pairs
    # csvPath is replaced with pairs which is a list of tuples where each tuple is (fixed_prot_id, mut_prot_id) 

    # modelPath = args.model
    # modelPath is replaced with model

    # outPath = args.outfile
    # outPath is not used to save prediction output in a file and instead it will be directly returned

    # seqPath = args.seqs
    # seqPath is replaced with seqs which is a list of tuples where each tuple is (prot_id, prot_seq) 

    # embPath = args.embeddings
    # embPath is not used as embeddings will be derived at runtime

    # device = args.device
    # device is not used as it is assumed to be cuda:0

    threshold = threshold
    use_cuda=True

    seqDict = {prot_id: prot_seq for prot_id, prot_seq in seqs}

    # print("Generating Embeddings...")
    embeddings = {}
    for n in seqDict.keys():
        embeddings[n] = lm_embed_pd(seqDict[n], use_cuda, pre_trn_mdl, alphabet)

    # Initialize the array with zeros
    dcp = np.zeros(len(pairs), dtype=float)
    # pred_prob_lst = []
    model.eval()
    with torch.no_grad():
        for idx, (n0, n1) in enumerate(pairs):
            p0 = embeddings[n0]
            p1 = embeddings[n1]

            if use_cuda:
                p0 = p0.cuda()
                p1 = p1.cuda()
            try:
                cm, p = model.map_predict(p0, p1)
                p = p.item()
                # ############## append prediction probability ##############
                # pred_prob_lst.append(p)
                dcp[idx] = 1.0 - p
            except RuntimeError as e:
                print(f"{n0} x {n1} skipped ({e})")
        # end of for loop: for n0, n1 in pairs:
    # end of with torch.no_grad():
    # return pred_prob_lst
    return dcp


def execute_prediction(i, pairs, model, seqDict, use_cuda  # Fixed data structures
                             , pred_prob_lst, shared_data_lock ): # Shared data structures
    n0, n1 = pairs[i]
    p0 = lm_embed(seqDict[n0], use_cuda)
    p1 = lm_embed(seqDict[n1], use_cuda)
    if use_cuda:
        p0 = p0.cuda()
        p1 = p1.cuda()
    try:
        cm, p = model.map_predict(p0, p1)
        p = p.item()
        # ############## append prediction probability ##############
        pred_prob_lst.append(p)
    except RuntimeError as e:
        print(f"{n0} x {n1} skipped ({e})")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
