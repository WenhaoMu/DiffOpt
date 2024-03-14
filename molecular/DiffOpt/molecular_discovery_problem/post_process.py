import os
import glob
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
from tdc import Oracle
from utils import z_to_smiles

def process_file(file_path, mean, std, SA_oracle, GSK3B_oracle, log_file, all_score):
    molecular = torch.load(file_path)
    molecular = molecular.detach().cpu()
    molecular = molecular * std + mean
    molecular = molecular.to(torch.float32)
    molecular = molecular.to('cuda')
    best_score = 0
    best_qed, best_sa, best_gsk3b = 0, 0, 0
    error_count_1, error_count_2, error_count_3 = 0, 0, 0

    log_file.write(f"File: {file_path}\n")
    print(f"File: {file_path}\n")
    for i in range(molecular.size(0)):
        a = molecular[i].unsqueeze(0)
        try:
            mol_smile = z_to_smiles(a)
            try:
                mol = Chem.MolFromSmiles(mol_smile[0])
                qed = QED.qed(mol)
                qed_nor = qed
                sa_score = SA_oracle(mol_smile[0])
                sa_score_nor = (sa_score - 1) / 9
                GSK3B_score = GSK3B_oracle(mol_smile[0])
                gsk3b_nor = GSK3B_score
                score = qed_nor - sa_score_nor + gsk3b_nor
                print(f'this is time {i}')
                if mol is None:
                    error_count_3 += 1
                    log_file.write('error3\n')
                else:
                    all_score.append((score, qed_nor, sa_score_nor, gsk3b_nor))
                    if score > best_score:
                        best_score = score
                        best_qed, best_sa, best_gsk3b = qed_nor, sa_score_nor, gsk3b_nor
                        log_file.write(f"Update at iteration {i}: Score - {score}, QED - {qed_nor}, SA - {sa_score_nor}, GSK3B - {gsk3b_nor}\n")
            except Exception as a:
                error_count_1 +=1
                log_file.write(f"Error in the {i+1} test: {a}\n")
        except Exception as e:
            error_count_2 += 1
            log_file.write(f"Error in the {i+1} test: {e}\n")

    log_file.write(f"File: {file_path}\n")
    log_file.write(f"Errors: {error_count_1, error_count_2, error_count_3}\n")
    log_file.write(f"Best score: {best_score}, QED: {best_qed}, SA: {best_sa}, GSK3B: {best_gsk3b}\n\n")
    log_file.flush()

def main():
    origin_data = np.load('../data/chembl_encoded_valid.npy')
    origin_value = np.load('../data/chembl_value_valid.npy')
    mean = origin_data.mean(axis=0)
    std = origin_data.std(axis=0)
    SA_oracle = Oracle(name = 'SA')
    GSK3B_oracle = Oracle(name = 'GSK3B')

    file_paths = glob.glob('./Result_DKL/*.pt')
    # file_paths = glob.glob('./Result_Final_Diff_Gui_Coefficient/molecular_gui_10000.0_256.pt')
    file_paths.sort(reverse=True) 
    
    with open('./Result_DKL/process_log.txt', 'a') as log_file:
        for file_path in file_paths:
            all_score = []
            process_file(file_path, mean, std, SA_oracle, GSK3B_oracle, log_file, all_score)

            top_scores = sorted(all_score, reverse=True)[:10]
            avg_score = sum([x[0] for x in top_scores]) / len(top_scores)
            avg_qed = sum([x[1] for x in top_scores]) / len(top_scores)
            avg_sa = sum([x[2] for x in top_scores]) / len(top_scores)
            avg_gsk3b = sum([x[3] for x in top_scores]) / len(top_scores)

            log_file.write(f"Average of Top 10 Scores: {avg_score}, QED: {avg_qed}, SA: {avg_sa}, GSK3B: {avg_gsk3b}\n")

if __name__ == "__main__":
    main()

