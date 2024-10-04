
import numpy as np
import torch
import io
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


from rdkit.Chem.Draw import rdMolDraw2D
import rdkit.Chem as Chem


from .MolAnalyzer import MolAnalyzer

class ISAAnalyzer(MolAnalyzer):
    """The class for additional tasks after the training. only for ISA type models"""
    def __init__(self, model_path, save_result = False):
        super().__init__(model_path , save_result)

        self.data_keys = ['prediction', 'positive', 'negative', 'feature_P', 'feature_N','fragments']
        self.for_pickle = ['fragments', ]
        if getattr(self.tm,'get_score',None) is None:
            raise ValueError("This model does not support ISAAnalyzer.")
        self.check_score_by_group()
    

    def check_score_by_group(self):
        if getattr(self.dm.gg,'sculptor',None) is None:
            self.is_score_by_group = True
        else:
            temp_smiles='CC(C)(C)OC(=O)C1=CC=CC=C1C(=O)O'
            
            test_loader,_ = self.prepare_temp_data([temp_smiles])
            temp_score = self.tm.get_score(self.nm, test_loader)

            if 'positive' in temp_score:
                temp_score['positive'] = temp_score['positive'].detach().cpu().numpy()
                if temp_score['positive'].shape[0] == len(self.get_fragment(temp_smiles)):
                    self.is_score_by_group = True
                elif temp_score['positive'].shape[0] == Chem.MolFromSmiles(temp_smiles).GetNumAtoms():
                    self.is_score_by_group = False
                else:
                    raise ValueError('The length of the score and the number of fragments do not match.')
            else:
                raise ValueError('The score is not calculated properly.')




    # get the attention score of the given smiles
    def get_score(self, smiles):
        "get the attention score of the given smiles by its subgroups"
        pos_score = self.load_data(smiles, 'positive')
        if pos_score is not None:
            neg_score = self.load_data(smiles, 'negative')
            if neg_score is not None:
                return {'positive': pos_score, 'negative': neg_score}
            else:
                return {'positive': pos_score}
        else:            
            test_loader,_ = self.prepare_temp_data([smiles])
            result = self.tm.get_score(self.nm, test_loader)
            for k in result.keys():
                if type(result[k]) is torch.Tensor:
                    result[k] = result[k].detach().cpu().numpy()

            result['positive'] = self.get_group_score(smiles, result['positive'])
            if 'negative' in result:
                result['negative'] = self.get_group_score(smiles, result['negative'])
            self.save_data(smiles, result)
            return result
    
    # get the fragment of the given smiles
    def get_fragment(self, smiles, get_index=False):
        frag = None#self.load_data(smiles, 'fragments')
        if frag is None:
            sculptor = self.dm.gg.sculptor
            frag = sculptor.fragmentation_with_condition(Chem.MolFromSmiles(smiles),draw=False,get_index = False)
            self.save_data(smiles, {'fragments': frag})
        if get_index:
            return [f.atoms for f in iter(frag)]
        return frag
    
    # get the hidden feature of the given smiles by its subgroups
    def get_feature(self, smiles):
        pos_score = self.load_data(smiles, 'feature_P')
        if pos_score is not None:
            neg_score = self.load_data(smiles, 'feature_N')
            if neg_score is not None:
                return {'feature_P': pos_score, 'feature_N': neg_score}
            else:
                return {'feature_P': pos_score}
        else:
            temp_loader,_ = self.prepare_temp_data([smiles])
            result = self.tm.get_feature(self.nm, temp_loader)
            for k in result.keys():
                if type(result[k]) is torch.Tensor:
                    result[k] = result[k].detach().cpu().numpy()

            result['feature_P'] = result['feature_P']
            if 'feature_N' in result:
                result['feature_N'] = result['feature_N']
            self.save_data(smiles, result)
            return result
        
    # get the hidden feature of the given list of smiles by its subgroups
    def get_features(self, smiles_list):
        results = {}
        new_smiles = []
        for smiles in smiles_list:
            result = {}
            pos_data = self.load_data(smiles,'feature_P')
            if pos_data is not None:
                result['feature_P'] = pos_data
                neg_data = self.load_data(smiles,'feature_N')
                if neg_data is not None:
                    result['feature_N'] = neg_data
            else:
                new_smiles.append(smiles)
                continue
            results[smiles] = result

        if len(new_smiles) > 0:
            valid_smiles, new_results = self._get_all_features(new_smiles)
            for smiles, result in zip(valid_smiles, new_results):
                results[smiles] = result
        return results
    
    
    def _get_all_features(self, smiles_list):
        temp_loader,valid_smiles = self.prepare_temp_data(smiles_list)
        features = self.tm.get_feature(self.nm, temp_loader)
        features = {k:features[k].detach().cpu().numpy() for k in features.keys()}
        results = []
        count=0
        for i, smiles in enumerate(valid_smiles['compound']):
            result = {}
            frag = self.get_fragment(smiles)
            result['feature_P'] = features['feature_P'][count:count+len(frag)]
            if 'feature_N' in features:
                result['feature_N'] =  features['feature_N'][count:count+len(frag)]
            count+=len(frag)
            results.append(result)
            self.save_data(smiles, result)
        return valid_smiles['compound'], results

    # get the attention score of the given list of smiles by its subgroups
    def get_scores(self, smiles_list):
        results = {}

        new_smiles = []
        for smiles in smiles_list:  
            result = {}
            pos_data = self.load_data(smiles,'positive')
            if pos_data is not None:
                result['positive'] = pos_data
                neg_data = self.load_data(smiles,'negative')
                if neg_data is not None:
                    result['negative'] = neg_data
            else:
                new_smiles.append(smiles)
                continue
            results[smiles] = result

        if len(new_smiles) > 0:
            valid_smiles, new_results = self._get_all_scores(new_smiles)
            for smiles, result in zip(valid_smiles, new_results):
                results[smiles] = result
        return results               
               
     
    def _get_all_scores(self, smiles_list):
        temp_loader,valid_smiles = self.prepare_temp_data(smiles_list)
        if len(valid_smiles) == 0:
            return [], []
        scores = self.tm.get_score(self.nm, temp_loader)
        scores = {k:scores[k].detach().cpu().numpy() for k in scores.keys()}
        results = []
        count=0
        for i, smiles in enumerate(valid_smiles['compound']):
            result={}
            result['prediction']=scores['prediction'][i]

            frag = self.get_fragment(smiles, get_index=True)
            result['positive'] = scores['positive'][count:count+len(frag)]
            if 'negative' in scores:
                result['negative'] = scores['negative'][count:count+len(frag)]
            count+=len(frag)
            results.append(result)
            
            
            result['positive'] = self.get_group_score(smiles, result['positive'],frag)
            if 'negative' in result:
                result['negative'] = self.get_group_score(smiles, result['negative'],frag)
            self.save_data(smiles, result)
        return valid_smiles['compound'], results
    
    def plot_score(self, smiles ,*, atom_with_index=False, score_scaler=lambda x:x, ticks= [0,0.25,0.5,0.75,1],
                   rot=0, line_width=2.0, locate='right',figsize=1, only_total=False, with_colorbar=True,):
        """
        This function plots the attention score of the given smiles by its subgroups.

        Args:
            smiles (str): The smiles of the molecule.
            atom_with_index (bool): Whether to show the atom index or not.
            score_scaler (function): The function to scale the score. Default is lambda x:x.
            ticks (list): The ticks of the colorbar. Default is [0,0.25,0.5,0.75,1].
            rot (int): The rotation of the molecule. Default is 0.
            locate (str): The location of the subplots. Default is 'right'. It can be 'right' or 'bottom'.
            figsize (float): The size of the figure. Default is 1.
            only_total (bool): Whether to plot only the total score or plot PAS and NAS as well. Default is False.
            with_colorbar (bool): Whether to show the colorbar or not. Default is True.

        Returns:
            dict: The attention score of the given smiles.
                  Each value in the dictionary is the attention score of corresponding atom with the same index.
        
        """
        score = self.get_score(smiles)
        mol = Chem.MolFromSmiles(smiles)
        
        colors = [(0, 'orangered'),(0.5,'white'),  (1, 'royalblue')]
        cmap_name = 'my_custom_colormap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors)

        def plot(ax,mol, score, title,figsize=1):
            png_bytes= showAtomHighlight(mol, score, cm, atom_with_index,rot,line_width,figsize=figsize)
            image = Image.open(io.BytesIO(png_bytes))
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(title, fontsize=20)
            
        _score_scaler = lambda x: score_scaler(x)#np.clip(,0,1)
                                                                             
        if 'positive' in score and 'negative' in score:
            score['positive'] = self.get_atom_score(smiles,score['positive'])
            score['negative'] = self.get_atom_score(smiles,score['negative'])

            pos_score = score['positive']
            neg_score = score['negative']
            tot_score = np.zeros_like(pos_score)
            tot_score=(1+pos_score-neg_score)/2
            tot_score = _score_scaler(tot_score)
            score['total'] = tot_score

            if only_total:
                fig, ax = plt.subplots(1, 1, figsize=(16, 11))
                plot(ax,mol, tot_score, ' ',figsize=figsize)
            else:
                if locate == 'right':
                    gs = GridSpec(2,3)
                    gs0 = gs[0:2, 0:2]
                    gs1 = gs[0, 2]
                    gs2 = gs[1, 2]
                    fig = plt.figure(figsize=(16*figsize, 10*figsize))
                elif locate == 'bottom':
                    gs = GridSpec(3,2)
                    gs0 = gs[0:2, 0:2]
                    gs1 = gs[2, 0]
                    gs2 = gs[2, 1]
                    fig = plt.figure(figsize=(12*figsize, 12*figsize))
                ax1 = fig.add_subplot(gs0)
                plot(ax1,mol, tot_score, ' ',figsize=figsize)
                ax2 = fig.add_subplot(gs1)
                plot(ax2,mol, pos_score, 'Positive',figsize=figsize*0.5)
                ax3 = fig.add_subplot(gs2)
                plot(ax3,mol, neg_score, 'Negative',figsize=figsize*0.5)
                plt.subplots_adjust(left=0, right=0.9, top=1, bottom=0)

                buf = io.BytesIO()
                plt.savefig(buf, format='png') 
                plt.close()
                buf.seek(0)  
                image = Image.open(buf)

                fig, ax = plt.subplots(1, 1, figsize=(16, 11))
                ax.imshow(image)
                ax.axis('off')
            if with_colorbar:
                cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cm), ax=ax, shrink=0.5, pad=0)
                cb.set_ticks([0,0.25,0.5,0.75,1])
                cb.set_ticklabels(ticks)
                cb.ax.tick_params(labelsize=20)
        else:
            score['positive'] = _score_scaler(score['positive'])
            score['positive'] = self.get_atom_score(smiles,score['positive'])  
            fig, ax = plt.subplots(1, 1, figsize=(16, 11))
            plot(ax,mol, score['positive'], ' ',figsize=figsize)
            if with_colorbar:
                cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cm), ax=ax, shrink=0.7)
                cb.set_ticks([0,0.25,0.5,0.75,1])
                cb.set_ticklabels(ticks)
                cb.ax.tick_params(labelsize=20)
        plt.show()
        return score
                    
    
    # get the bin of the attention score of the given list of smiles by its subgroups
    def get_subgroup_score_bin(self,smiles_list):
        results = self.get_scores(smiles_list)

        positive_frag_scores = {}
        negative_frag_scores = {}
        for smiles in results:
            score = results[smiles]
            if 'positive' in score and score['positive'].ndim != 2:
                score['positive'] = score['positive'].reshape(-1,1)
            if 'negative' in score and score['negative'].ndim != 2:
                score['negative'] = score['negative'].reshape(-1,1)
            frag = self.get_fragment(smiles)

            for i,f in enumerate(frag):
                f_smiles = f.smiles
                if 'positive' in score:
                    if f_smiles not in positive_frag_scores:
                        positive_frag_scores[f_smiles] = score['positive'][i]
                    else:
                        positive_frag_scores[f_smiles] = np.concatenate([positive_frag_scores[f_smiles], score['positive'][i]], axis=0)
                if 'negative' in score:
                    if f_smiles not in negative_frag_scores:
                        negative_frag_scores[f_smiles] = score['negative'][i]
                    else:
                        negative_frag_scores[f_smiles] = np.concatenate([negative_frag_scores[f_smiles], score['negative'][i]], axis=0)
        return positive_frag_scores, negative_frag_scores
    
    # plot the histogram of the attention score of the given list of smiles without regarding the subgroups
    def plot_score_hist(self, smiles_list,xlim=[0,1], bins=40):
        """
        This function plots the histogram of the attention score of the given list of smiles without regarding the subgroups.

        Args:
            smiles_list (list): The list of smiles.
            xlim (list): The range of the x-axis. Default is [0,1].
            bins (int): The number of bins. Default is 40.

        Returns:
            list: The list of PAS
            list: The list of NAS
        
        """
        results = self.get_scores(smiles_list)
        pos_scores = []
        neg_scores = []
        for smiles in results:
            score = results[smiles]
            if 'positive' in score:
                pos_scores+=set(list(score['positive'].flatten()))
            if 'negative' in score:
                neg_scores+=set(list(score['negative'].flatten()))
        
        if len(neg_scores) > 0:
            plt.hist(pos_scores, bins=np.arange(xlim[0], xlim[1], 1./bins), alpha=0.5, label='positive')
            plt.hist(neg_scores, bins=np.arange(xlim[0], xlim[1], 1./bins), alpha=0.5, label='negative')
            plt.legend()
        else:
            plt.hist(pos_scores, bins=np.arange(xlim[0], xlim[1], 1./bins), alpha=1, label='positive')
        plt.xlim(xlim[0], xlim[1])
        plt.show()
        
        return pos_scores, neg_scores
    
    # plot the histogram of the attention score of the given list of smiles by its subgroups
    def plot_subgroup_score_histogram(self, smiles_list, nums = 10, bins=40, xlim=[0,1]):
        """
        This function plots the histogram of the attention score of the given list of smiles by its subgroups.

        Args:
            smiles_list (list): The list of smiles.
            nums (int): The number of subgroups to plot. Default is 10.
                        The most frequent subgroups will be plotted.
            bins (int): The number of bins. Default is 40.
            xlim (list): The range of the x-axis. Default is [0,1].
        
        Returns:
            dict: PAS of subgroups 
            dict: NAS of subgroups 
        
        """
        pos_frag, neg_frag = self.get_subgroup_score_bin(smiles_list)
        
        pos_frag = dict(sorted(pos_frag.items(), key=lambda x: len(x[1]), reverse=True)[:nums])
        for i,f in enumerate(pos_frag):
            plt.hist(pos_frag[f], bins=np.arange(xlim[0], xlim[1], 1./bins), alpha=0.5, label=f)
        plt.xlim(xlim[0], xlim[1])
        plt.legend()
        plt.title('Positive')
        plt.show()

        if len(neg_frag) == 0:
            return pos_frag, neg_frag
        
        neg_frag = dict(sorted(neg_frag.items(), key=lambda x: len(x[1]), reverse=True)[:nums])
        for i,f in enumerate(neg_frag):
            plt.hist(neg_frag[f], bins=np.arange(xlim[0], xlim[1], 1./bins), alpha=0.5, label=f)
        plt.xlim(xlim[0], xlim[1])
        plt.legend()
        plt.title('Negative')
        plt.show()
        return pos_frag, neg_frag
        
    # plot the histogram of the attention score of the given list of smiles by its subgroups by one
    def plot_subgroup_score_histogram_byone(self, smiles_list, nums = 10, bins=40, xlim=[0,1], target_subgroups= None):
        """
        This function plots the histogram of the attention score of the given list of smiles by its subgroups by one.

        Args:
            smiles_list (list): The list of smiles.
            nums (int): The number of subgroups to plot. Default is 10.
                        The most frequent subgroups will be plotted.
            bins (int): The number of bins. Default is 40.
            xlim (list): The range of the x-axis. Default is [0,1].
            target_subgroups (list): The list of subgroups to plot. Default is None.
                                     If it is not None, only the subgroups in the list will be plotted.

        Returns:
            dict: PAS of subgroups 
            dict: NAS of subgroups 
        
        """
        pos_frag, neg_frag = self.get_subgroup_score_bin(smiles_list)
        
        pos_frag = dict(sorted(pos_frag.items(), key=lambda x: len(x[1]), reverse=True)[:nums])
        if len(neg_frag) > 0:
            neg_frag = dict(sorted(neg_frag.items(), key=lambda x: len(x[1]), reverse=True)[:nums])
        else:
            neg_frag = None
        
        if target_subgroups:
            pos_frag = {k:pos_frag[k] for k in pos_frag if k in target_subgroups}
            if neg_frag:
                neg_frag = {k:neg_frag[k] for k in neg_frag if k in target_subgroups}
            nums = len(pos_frag)

        fig,axs = plt.subplots(nums,1, figsize=(5, 2*nums))
        for i,f in enumerate(pos_frag):
            axs[i].hist(pos_frag[f], bins=np.arange(xlim[0], xlim[1], 1./bins), alpha=0.5, label='positive')
            if neg_frag:
                axs[i].hist(neg_frag[f], bins=np.arange(xlim[0], xlim[1], 1./bins), alpha=0.5, label='negative')
                axs[i].set_ylabel(f)
            axs[i].set_xlim(xlim[0], xlim[1])
            axs[i].legend()
        plt.show()
        return pos_frag, neg_frag

    def get_atom_score(self, smiles, score,frag=None):
        mol = Chem.MolFromSmiles(smiles)
        if len(score) < mol.GetNumAtoms():
            score = self.group_score2atom_score(smiles, score, frag)
        if len(score) != mol.GetNumAtoms():
            raise ValueError('The length of score and the number of atoms do not match.')
        return score
    
    def get_group_score(self, smiles, score, frag=None):
        if self.is_score_by_group:
            return score
        if frag is None:
            frag = self.get_fragment(smiles, get_index=True)
        if len(frag) < score.shape[0]:
            score = self.atom_score2group_score(smiles, score, frag)
        if len(frag) != score.shape[0]:
            raise ValueError('The number of fragments and the length of score do not match.')
        return score

    def group_score2atom_score(self, smiles, group_score, frag=None):
        mol= Chem.MolFromSmiles(smiles)
        if frag is None:
            frag = self.get_fragment(smiles, get_index=True)
        if len(frag)!=group_score.shape[0]:
            raise ValueError('The number of fragments and the length of group_score do not match.')
        atom_score = [0]*mol.GetNumAtoms()
        for f,score in zip(frag,group_score):
            for a in f:
                atom_score[a]=score
        atom_score = np.array(atom_score)
        return atom_score
    
    def atom_score2group_score(self, smiles, atom_score, frag=None):
        if frag is None:
            frag = self.get_fragment(smiles, get_index=True)
        if len(atom_score) != sum([len(f) for f in frag]):
            raise ValueError('The length of atom_score and the number of atoms do not match.')
        group_score = []
        for f in frag:
            score = atom_score[f[0]]
            group_score.append(score)
        group_score = np.array(group_score)
        return group_score

    
    # def _plot_feature(self,smiles_list, highlight=None, label=None):
    #     feats_result = self.get_features(smiles_list)
    #     score_result = self.get_scores(smiles_list)
    #     frags_list = {}
    #     frags_name_list = {}
    #     for smiles in smiles_list:
    #         frags_list[smiles]=[f.smiles if f.name != 'Phenyl' else '*c1ccccc1' for f in self.get_fragment(smiles, get_index=False)]
    #         frags_name_list[smiles]=[f.name for f in self.get_fragment(smiles, get_index=False)]
    #     feats= {'feature_P':[],'feature_N':[],'positive':[],'negative':[],'fragments':[],'smiles':[]}
    #     for smiles in feats_result:
    #         if smiles not in score_result.keys():
    #             continue
    #         for k in feats.keys():
    #             if not (smiles in feats_result and smiles in score_result):
    #                 continue
    #             if k in feats_result[smiles]:
    #                 feats[k].append(feats_result[smiles][k])
    #             if k in score_result[smiles]:
    #                 feats[k].append(score_result[smiles][k])
    #         feats['fragments'].append(frags_list[smiles])
    #         feats['smiles'].append([smiles]*len(frags_list[smiles]))

    #         for i in range(len(frags_list[smiles])):
    #             if frags_list[smiles][i] =="c1ccccc1" and score_result[smiles]['positive'][i]>0.7:
    #                 print(smiles, score_result[smiles]['positive'][i])
    #     return 
    #     for k in feats.keys():
    #         feats[k] = np.concatenate(feats[k],axis=0)

    #     import umap
    #     fit = umap.UMAP(
    #         n_neighbors=50,
    #         min_dist=.1,
    #         n_components=2,
    #         metric='euclidean'
    #     )
    #     reduced_data = fit.fit_transform(feats['feature_P'])
    #     fig = plt.figure(figsize=(6,6))
    #     ax = fig.add_subplot(111)
    #     ax.scatter(reduced_data[:,0],reduced_data[:,1],c = feats['positive'],alpha=0.3,s=4)
    #     plt.show()


    #     from sklearn.manifold import TSNE
    #     from matplotlib.patches import Patch

    #     if highlight:
    #         col_p = np.zeros((feats['positive'].shape[0],4))
    #         col_p[:]=0.85
    #         col_n = np.zeros((feats['positive'].shape[0],4))
    #         col_n[:]=0.85
    #         if type(highlight) is str:
    #             highlight = [highlight]
    #         cmap= plt.cm.tab20
    #         legend_list = []
    #         for i,h in enumerate(highlight):
    #             if i>=15: j=i+1
    #             else: j=i
    #             col_p[feats['fragments']==h] = cmap(j)
    #             col_n[feats['fragments']==h] = cmap(j)
    #             legend_list.append(Patch(facecolor=cmap(j), edgecolor='black',label=label[i] if label else h))
    #         legend_list.append(Patch(facecolor='lightgray', edgecolor='black',label='others'))
    #     else:
    #         col_p = (1-feats['positive']).flatten()
    #         col_n = (1-feats['negative']).flatten()
    #         cmap='bwr'

    
    #     tsne_p = TSNE(n_components=2,perplexity=20., early_exaggeration=8., random_state=42) 
    #     reduced_data = tsne_p.fit_transform(feats['feature_P'])
    #     fig = plt.figure(figsize=(6,6))
    #     ax = fig.add_subplot(111)
    #     if highlight:
    #         ax.legend(handles=legend_list,loc = 'center left', bbox_to_anchor=(1, 0.5))
    #     ax.scatter(reduced_data[:,0],reduced_data[:,1],c = col_p, alpha=0.3,s=3)
    #     ax.set_axis_off()
    #     ax.set_title('Positive')
    #     plt.show()

    #     tsne_n = TSNE(n_components=2,perplexity=20., early_exaggeration=8., random_state=42) 
    #     reduced_data = tsne_n.fit_transform(feats['feature_N'])


    #     fig = plt.figure(figsize=(6,6))
    #     ax = fig.add_subplot(111)
    #     if highlight:
    #         ax.legend(handles=legend_list,loc = 'center left', bbox_to_anchor=(1, 0.5))
    #     ax.scatter(reduced_data[:,0],reduced_data[:,1],c = col_n, alpha=0.3,s=3)
    #     ax.set_axis_off()
    #     ax.set_title('Negative')
    #     plt.show()

    # def _analyze_synergy(self,smiles,STATE='positive'):
    #     g= self.dm.gg.get_graph(smiles)
    #     frag = self.get_fragment(smiles, get_index=True)
    #     def subgroup_combination(g):
    #         from itertools import combinations
    #         n = g.num_nodes('d_nd')
    #         numbers = list(range(n))
    #         all_combinations = []
    #         for r in range(len(numbers) + 1):
    #             all_combinations.extend(combinations(numbers, r))

    #         sub_comb= []
    #         remains = []
    #         for i in all_combinations:
    #             if len(i) == n: continue
    #             _g=g.clone()
    #             _g.remove_nodes(i,ntype='d_nd')
    #             mask = []
    #             for k in i:
    #                 for f in frag[k]:
    #                     mask.append(f)
    #             _g.remove_nodes(mask,ntype='i_nd')
    #             _g.remove_nodes(mask,ntype='r_nd')
    #             sub_comb.append(_g)
    #             result = set(numbers)-set(i)
    #             result = list(result)
    #             result.sort()
    #             remains.append(tuple(result))
    #         return sub_comb, remains


    #     gs,remains = subgroup_combination(g)
    #     self.dm.init_temp_data([smiles]*len(gs),graphs = gs)
    #     test_loader = self.dm.get_temp_Dataloader()
    #     scores = self.tm.get_score(self.nm, test_loader)

    #     count=0
    #     results = {}
    #     def reorder_lists(lst):
    #         flattened = [num for sublist in lst for num in sublist]
    #         sorted_numbers = sorted(flattened)
            
    #         reordered = []
    #         for sublist in lst:
    #             reordered.append([sorted_numbers.index(i) for i in sublist])
            
    #         return reordered
        
    #     for i in range(len(gs)):
    #         result={}
    #         result['prediction']=scores['prediction'][i].detach().cpu().numpy()
    #         remain_frag = reorder_lists([frag[j] for j in remains[i]])
    #         if not self.is_score_by_group:
    #             atom_num = sum([len(f) for f in remain_frag])
    #             result[STATE] = self.atom_score2group_score(smiles,scores[STATE][count:count+atom_num].detach().cpu().numpy(),remain_frag)
    #             count+=atom_num
    #         else:
    #             result[STATE] = scores[STATE][count:count+len(remain_frag)].detach().cpu().numpy()
    #             count+=len(frag)
    #         results[remains[i]] = result
            
    #     from math import factorial

    #     synergy_matrix = np.zeros((len(frag),len(frag)))
    #     for i in range(len(frag)):
    #         for j in range(len(frag)):
    #             if i==j:
    #                 _remain = [r for r in remains if i in r]
    #                 val= np.sum([factorial(len(r)-1)*factorial(len(frag)-len(r))/factorial(len(frag))*results[r][STATE][r.index(i)] for r in _remain])
    #                 synergy_matrix[i][j] = val
    #             else:
    #                 i_remain = [r for r in remains if j not in r and i in r]
    #                 vals = []
    #                 for r in i_remain:
    #                     ij_remain = list(r)
    #                     ij_remain.append(j)
    #                     ij_remain.sort()
    #                     ij_remain = tuple(ij_remain)
    #                     vals.append(factorial(len(r))*factorial(len(frag)-len(r)-1)/factorial(len(frag))*(results[ij_remain][STATE][ij_remain.index(i)]-results[r][STATE][r.index(i)]))
    #                 val = np.sum(vals)
    #                 synergy_matrix[i][j] = val

    #     shaps = np.zeros(len(frag))
    #     for i in range(len(frag)):
    #         _remain = [r for r in remains if i not in r]
    #         valse = []
    #         for r in _remain:
    #             i_remain = list(r)
    #             i_remain.append(i)
    #             i_remain.sort()
    #             i_remain = tuple(i_remain)
    #             valse.append(factorial(len(r))*factorial(len(frag)-len(r)-1)/factorial(len(frag))*(results[i_remain]['prediction']-results[r]['prediction']) )
    #         shaps[i] = np.sum(valse)
            
    #     min_value = np.min(synergy_matrix - 0.5 * np.eye(len(frag)))
    #     max_value = np.max(synergy_matrix - 0.5 * np.eye(len(frag)))
    #     top = mpl.colormaps.get_cmap('Oranges_r')
    #     bottom = mpl.colormaps.get_cmap('Blues')

    #     synergy_matrix=synergy_matrix.T
    #     newcolors = np.vstack((top(np.linspace(0, 1, np.abs(int(min_value*200)))),
    #                         bottom(np.linspace(0, 1, np.abs(int(max_value*200))))))
    #     newcmp = ListedColormap(newcolors, name='OrangeBlue')
    #     fig, ax = plt.subplots()
    #     fig.set_size_inches(0.93*len(synergy_matrix),0.9*len(synergy_matrix))
    #     im = ax.imshow(synergy_matrix-0.5*np.eye(len(frag)),cmap=newcmp)
    #     for i in range(synergy_matrix.shape[0]):
    #         for j in range(synergy_matrix.shape[1]):
    #             if np.abs(synergy_matrix[i, j]) <= 0.01:
    #                 continue
    #             text = ax.text(j, i, f"{synergy_matrix[i, j]:.2f}",
    #                         ha="center", va="center", color="w",fontsize=16)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     plt.show()
    #     # return frag, synergy_matrix, shaps

    #     transform = np.zeros((5,5))
    #     transform[0,0] = 1
    #     transform[1,4] = 1
    #     transform[2,3] = 1
    #     transform[3,2] = 1
    #     transform[4,1] = 1
    #     synergy_matrix= np.dot(np.dot(transform,synergy_matrix),transform.T)
    #     newcolors = np.vstack((top(np.linspace(0, 1, np.abs(int(min_value*200)))),
    #                         bottom(np.linspace(0, 1, np.abs(int(max_value*200))))))
    #     newcmp = ListedColormap(newcolors, name='OrangeBlue')
    #     fig, ax = plt.subplots()
    #     fig.set_size_inches(0.93*len(synergy_matrix),0.9*len(synergy_matrix))
    #     im = ax.imshow(synergy_matrix-0.5*np.eye(len(frag)),cmap=newcmp)
    #     for i in range(synergy_matrix.shape[0]):
    #         for j in range(synergy_matrix.shape[1]):
    #             if np.abs(synergy_matrix[i, j]) <= 0.01:
    #                 continue
    #             text = ax.text(j, i, f"{synergy_matrix[i, j]:.2f}",
    #                         ha="center", va="center", color="w",fontsize=16)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     plt.show()

    #     return frag, synergy_matrix, shaps

            





def showAtomHighlight(mol,score,color_map,atom_with_index=True,rot=0,line_width=1.0,figsize=1):
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)
    atom_num = mol.GetNumAtoms()

    if atom_with_index:
        mol=mol_with_atom_index(mol)
    hit_ats=[]
    hit_bonds = []
    hit_ats_colormap={}
    hit_bonds_colormap={}
    

    for i in range(atom_num):
        hit_ats.append(i)
        hit_ats_colormap[i] = color_map(score[i].item())
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        
        if score[src] == score[dst]:
            hit_bonds.append(i)
            hit_bonds_colormap[i] = color_map(score[src].item())
                    

    d = rdMolDraw2D.MolDraw2DCairo(int(1000*figsize), int(800*figsize))
    dopts = d.drawOptions()
    dopts.rotate = rot
    dopts.bondLineWidth = line_width
    d.DrawMolecule(mol,highlightAtoms = hit_ats, highlightAtomColors=hit_ats_colormap,
                   highlightBonds=hit_bonds, highlightBondColors=hit_bonds_colormap)

    d.FinishDrawing()
    return d.GetDrawingText()


def mol_with_atom_index( mol ):
    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol
