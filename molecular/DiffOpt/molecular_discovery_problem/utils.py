import subprocess
from tqdm import tqdm
import os
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import MolFromSmiles, QED
from rdkit.Chem import AllChem
from sascorer import calculateScore
import numpy as np
import time
import math
import json
import torch
import csv
from hgraph2graph.hgraph import *
import argparse
import h5py
from sklearn.decomposition import PCA


delta_g_to_kd = lambda x: math.exp(x / (0.00198720425864083 * 298.15))
kd_to_delta_g = lambda x: 0.00198720425864083 * 298.15 * math.log(x)

# f = h5py.File('/data/peter/for_allen/target_embeddings.h5', 'r')
# target_to_embedding = {}
# targets = []
# x = []
# for key in f:
#     target = key.split('_')[0]
#     targets.append(target)
#     x.append(np.array(f[key]))
# x = np.array(x)
# x = PCA(64).fit_transform(x)
# np.save('target_embeddings.npy', x)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', default='hgraph2graph/data/chembl/vocab.txt')
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', default='hgraph2graph/ckpt/chembl-pretrained/model.ckpt')

parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--nsample', type=int, default=10000)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

args = parser.parse_args()

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)

model = HierVAE(args).cuda()

model.load_state_dict(torch.load(args.model)[0])
model.eval()

    
def smiles_to_sa(smiles):
    vals = []
    for smile in tqdm(smiles):
        vals.append(calculateScore(MolFromSmiles(smile)))
    return vals


def smiles_to_hac(smiles):
    vals = []
    for smile in tqdm(smiles):
        vals.append(MolFromSmiles(smile).GetNumHeavyAtoms())
    return vals


def smiles_to_qed(smiles):
    vals = []
    for smile in tqdm(smiles):
        vals.append(QED.qed(MolFromSmiles(smile)))
    return vals


def smiles_to_logp(smiles):
    vals = []
    for smile in tqdm(smiles):
        vals.append(MolLogP(MolFromSmiles(smile)))
    return vals


def smiles_to_morgan(smiles):
    out = []
    for smile in tqdm(smiles):
        out.append(np.array(AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(smile), 3, nBits=2048)))
    return np.array(out)


def smiles_to_affinity(smiles, autodock='~/AutoDock-GPU/bin/autodock_gpu_128wi', protein_file='/data/peter/pdbs/b65d866690c8eaf8.maps.fld', num_devices=torch.cuda.device_count(), starting_device=0):
    if not os.path.exists('ligands'):
        os.mkdir('ligands')
    if not os.path.exists('outs'):
        os.mkdir('outs')
    subprocess.run('rm core.*', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.xml', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.dlg', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm -rf ligands/*', shell=True, stderr=subprocess.DEVNULL)
    for device in range(starting_device, starting_device + num_devices):
        os.mkdir(f'ligands/{device}')
    device = starting_device
    for i, smile in enumerate(tqdm(smiles, desc='preparing ligands')):
        subprocess.Popen(f'obabel -:"{smile}" -O ligands/{device}/ligand{i}HASH{hash(smile)}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d', shell=True, stderr=subprocess.DEVNULL)
        device += 1
        if device == starting_device + num_devices:
            device = starting_device
    while True:
        total = 0
        for device in range(starting_device, starting_device + num_devices):
            total += len(os.listdir(f'ligands/{device}'))
        if total == len(smiles):
            break
    time.sleep(1)
    print('running autodock..')
    subprocess.run('rm outs/*.xml', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.dlg', shell=True, stderr=subprocess.DEVNULL)
    if len(smiles) == 1:
        subprocess.run(f'{autodock} -M {protein_file} -L ligands/0/ligand0.pdbqt -N outs/ligand0', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        ps = []
        for device in range(starting_device, starting_device + num_devices):
            ps.append(subprocess.Popen(f'{autodock} -M {protein_file} -B ligands/{device}/ligand*.pdbqt -N ../../outs/ -D {device + 1}', shell=True, stdout=subprocess.DEVNULL))
        stop = False
        while not stop: 
            stop = True
            for p in ps:
                if p.poll() is None:
                    time.sleep(1)
                    stop = False
    affins = [0 for _ in range(len(smiles))]
    for file in os.listdir('outs'):
        if file.endswith('.dlg'):
            content = open(f'outs/{file}').read()
            if '0.000   0.000   0.000  0.00  0.00' not in content:
                try:
                    affins[int(file.split('ligand')[1].split('HASH')[0])] = float([line for line in content.split('\n') if 'RANKING' in line][0].split()[3])
                except:
                    pass
    return [min(affin, 0) for affin in affins]


def get_target_embeddings(ts):
    out = []
    for t in ts:
        out.append(x[targets.index(t)])
    return np.array(out)


def autodock(smiles, target):
    affins = np.array(smiles_to_affinity(smiles * 10, protein_file=f'/data/peter/for_allen/all/{target}/receptor.maps.fld'))
    affin_mins = affins.reshape((-1, len(smiles))).min(0)
    smile_to_data = {smiles[i]: {'total_energy': affin_mins[i]} for i in range(len(smiles))}
    for smile in smile_to_data:
        for f_name in [f for f in os.listdir('outs') if ('.dlg' in f and str(hash(smile)) in f)]:
            f = open(f'outs/{f_name}', 'r').read()
            if f"Estimated Free Energy of Binding    =  {smile_to_data[smile]['total_energy']}" in f or f"Estimated Free Energy of Binding    = {smile_to_data[smile]['total_energy']}" in f:
                if f"Estimated Free Energy of Binding    =  {smile_to_data[smile]['total_energy']}" in f:
                    pdb = f.split(f"Estimated Free Energy of Binding    =  {smile_to_data[smile]['total_energy']}")[1]
                else:
                    pdb = f.split(f"Estimated Free Energy of Binding    = {smile_to_data[smile]['total_energy']}")[1]
                smile_to_data[smile]['intermolecular_energy'] = float(f.split('(1) Final Intermolecular Energy')[1].split()[1].strip())
                smile_to_data[smile]['internal_energy'] = float(f.split('(2) Final Total Internal Energy')[1].split()[1].strip())
                smile_to_data[smile]['torsional_energy'] = float(f.split('(3) Torsional Free Energy')[1].split()[1].strip())
                smile_to_data[smile]['unbound_energy'] = float(f.split('(4) Unbound System\'s Energy')[1].split()[1].strip())
                pdb = pdb.split('DOCKED: REMARK                         _______ _______ _______ _____ _____    ______ ____')[1].split('DOCKED: ENDMDL')[0].strip()
                pdb = pdb.replace('DOCKED: ', '')
                atoms = []
                for line in pdb.split('\n'):
                    if line.startswith('ATOM'):
                        _, _, type, _, _, x, y, z, vdw, _, _, _ = line.split()
                        atoms.append((type, float(vdw), float(x), float(y), float(z)))
                smile_to_data[smile]['atom_coords'] = atoms
                smile_to_data[smile]['number_of_atoms'] = int(f.split('Number of atoms:')[1].split()[0].strip())
                smile_to_data[smile]['number_of_rotatable_bonds'] = int(f.split('Number of rotatable bonds:')[1].split()[0].strip())
                pdb += '\nENDMDL'
                new_pdb = ''
                for line in pdb.split('\n'):
                    new_pdb += line[:66] + '\n'
                pdb = new_pdb
                open('autodock_pose.pdb', 'w').write(pdb)
                subprocess.run('pymol -cq pymol_script.py', shell=True)
                subprocess.call(f'obabel autodock_pose.pdb -O autodock_pose.pdbqt -p 7.4', shell=True, stdout=subprocess.DEVNULL)
                subprocess.call('python /home/peter/limo/binana/python/run_binana.py -receptor /data/peter/pdbs/b65d866690c8eaf8.pdbqt -ligand autodock_pose.pdbqt -output_json binana_out.json', shell=True, stdout=subprocess.DEVNULL)
                out = json.load(open('binana_out.json', 'r'))
                smile_to_data[smile]['hydrogen_bonds'] = len(out['hydrogenBonds'])
                smile_to_data[smile]['pi_pi_stacking_interactions'] = len(out['piPiStackingInteractions'])
                smile_to_data[smile]['salt_bridges'] = len(out['saltBridges'])
                smile_to_data[smile]['t_stacking_interactions'] = len(out['tStackingInteractions'])
                break
    return smile_to_data


def abfe(smiles):
    pass


def load_bindingdb_data(file):
    out = []
    for row in csv.reader(open(file, 'r'), delimiter='	'):
        if row[9] and '<' not in row[9] and '>' not in row[9] and row[9] != 'IC50 (nM)':
            out.append([row[1], kd_to_delta_g(float(row[9]) / 1e9)])
    return out


def dock6_flex(smiles):
    smile_to_data = {}
    for start_index in range(0, len(smiles), proc_max):
        subprocess.run('rm -rf dock6/runs', shell=True, stderr=subprocess.DEVNULL)
        os.mkdir('dock6/runs')
        ps = []
        for i, smile in enumerate(smiles[start_index:start_index + proc_max]):
            dir = f'dock6/runs/{i}'
            os.mkdir(dir)
            ps.append(subprocess.Popen(f'obabel -:"{smile}" -O {dir}/ligand.mol2 -h --partialcharge mmff94 --gen3d', shell=True))
        for p in ps:
            p.wait()
        open('run_dock6_flex_modified', 'w').write(open('run_dock6_flex', 'r').read().replace('_MAX_FILE_', str(len(smiles[start_index:start_index + proc_max]) - 1)))
        subprocess.run(f'./run_dock6_flex_modified')
        for i, smile in enumerate(smiles[start_index:start_index + proc_max]):
            res = open(f'dock6/runs/{i}/flex.out').read().split('Secondary Score')[1]
            smile_to_data[smile] = {'total_score': float(res.split('Grid_Score:')[1].split()[0].strip()),
                                    'vdw_score': float(res.split('Grid_vdw_energy:')[1].split()[0].strip()),
                                    'es_score': float(res.split('Grid_es_energy:')[1].split()[0].strip()),
                                    'internal_energy': float(res.split('Internal_energy_repulsive:')[1].split()[0].strip())}
            subprocess.call(f'obabel dock6/runs/{i}/dock1_secondary_scored.mol2 -O dock6/runs/{i}/out.pdb', shell=True)
            atoms = []
            for line in open(f'dock6/runs/{i}/out.pdb', 'r'):
                if line.startswith('ATOM'):
                    _, _, type, _, _, x, y, z, vdw, _, _, _ = line.split()
                    atoms.append((type, float(x), float(y), float(z)))
            smile_to_data[smile]['atom_coords'] = atoms
    return smile_to_data


def dock6_amber(smiles):
    smile_to_data = {}
    for start_index in range(0, len(smiles), proc_max):
        subprocess.run('rm -rf dock6/runs', shell=True, stderr=subprocess.DEVNULL)
        os.mkdir('dock6/runs')
        ps = []
        for i, smile in enumerate(smiles[start_index:start_index + proc_max]):
            dir = f'dock6/runs/{i}'
            os.mkdir(dir)
            ps.append(subprocess.Popen(f'obabel -:"{smile}" -O {dir}/ligand.mol2 -h --partialcharge mmff94 --gen3d', shell=True))
        for p in ps:
            p.wait()
        open('run_dock6_amber_modified', 'w').write(open('run_dock6_amber', 'r').read().replace('_MAX_FILE_', str(len(smiles[start_index:start_index + proc_max]) - 1)))
        subprocess.run(f'./run_dock6_amber_modified')
        for i, smile in enumerate(smiles[start_index:start_index + proc_max]):
            res = open(f'dock6/runs/{i}/amber.out').read().split('Conformations:')[1]
            smile_to_data[smile] = {'amber_score': float(res.split('Amber_Score:')[1].split()[0].strip()),
                                    'complex_energy': float(res.split('Amber_complex_energy:')[1].split()[0].strip()),
                                    'receptor_energy': float(res.split('Amber_receptor_energy:')[1].split()[0].strip()),
                                    'ligand_energy': float(res.split('Amber_ligand_energy:')[1].split()[0].strip())}
            atoms = []
            for line in open(f'dock6/runs/{i}/5uf0_noh.dock1_secondary_scored.1.final_pose.amber.pdb', 'r'):
                if line.startswith('ATOM'):
                    _, _, type, _, _, x, y, z, vdw, _ = line.split()
                    for char in '0123456789':
                        type = type.replace(char, '')
                    if type not in ['CL', 'BR']:
                        type = type[0]
                    atoms.append((type, float(x), float(y), float(z)))
            smile_to_data[smile]['receptor_atom_coords'] = atoms
            atoms = []
            for line in open(f'dock6/runs/{i}/dock1_secondary_scored.1.final_pose.amber.pdb', 'r'):
                if line.startswith('ATOM'):
                    _, _, type, _, _, x, y, z, vdw, _ = line.split()
                    for char in '0123456789':
                        type = type.replace(char, '')
                    if type not in ['CL', 'BR']:
                        type = type[0]
                    atoms.append((type, float(x), float(y), float(z)))
            smile_to_data[smile]['ligand_atom_coords'] = atoms
    return smile_to_data


def z_to_smiles(z):
    return model.decoder.decode((z, z, z), greedy=True, max_decode_step=150)


def get_objectives(z, targets):
    z = z.view((-1, 32))
    out = []
    for i in range(len(z)):
        try:
            target = targets[i]
            smiles = z_to_smiles(z[i].view((1, 32)))
            autodock_results = autodock(smiles, target)
            hac = smiles_to_hac(smiles)
            logp = smiles_to_logp(smiles)
            qed = smiles_to_qed(smiles)
            sa = smiles_to_sa(smiles)
            total_energy = [autodock_results[smile]['total_energy'] for smile in smiles]
            torsional_energy = [autodock_results[smile]['torsional_energy'] for smile in smiles]
            hydrogen_bonds = [autodock_results[smile]['hydrogen_bonds'] for smile in smiles]
            pi_pi_stacking_interactions = [autodock_results[smile]['pi_pi_stacking_interactions'] for smile in smiles]
            salt_bridges = [autodock_results[smile]['salt_bridges'] for smile in smiles]
            t_stacking_interactions = [autodock_results[smile]['t_stacking_interactions'] for smile in smiles]
            ligand_efficiency = [total_energy[i] / hac[i] for i in range(len(smiles))]
            out.append({'logp': logp[0],
                        'qed': qed[0],
                        'sa': sa[0],
                        'total_energy': total_energy[0],
                        'torsional_energy': torsional_energy[0],
                        'hydrogen_bonds': hydrogen_bonds[0],
                        'pi_pi_stacking_interactions': pi_pi_stacking_interactions[0],
                        'salt_bridges': salt_bridges[0],
                        't_stacking_interactions': t_stacking_interactions[0],
                        'ligand_efficiency': ligand_efficiency[0]})
        except:
            out.append({'smiles': '',
                        'logp': 0,
                        'qed': 0,
                        'sa': 0,
                        'total_energy': 0,
                        'torsional_energy': 0,
                        'hydrogen_bonds': 0,
                        'pi_pi_stacking_interactions': 0,
                        'salt_bridges': 0,
                        't_stacking_interactions': 0,
                        'ligand_efficiency': 0})
    return out