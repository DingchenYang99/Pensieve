import json
from tqdm import tqdm
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from spice.spice import Spice
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.whoops.whoops_utils import *

import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import matplotlib.pyplot as plt

class WhoopsEvalCap:
    def __init__(self, results, gts, data=['all']):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.results = results  # {image_id:["caption":], ...}
        self.gts = gts  # {image_id:[{"caption":}, {"caption":}, ...], ...}
        if data == ['all']:
            self.params = {'image_id': gts.keys()}
        else:
            self.params = {'image_id': list(set(data))}
        
    def evaluate(self):
        imgIds = self.params['image_id']
        gts = {}
        res = {}
        print(f"evaluating {len(imgIds)} qsamples")
        for imgId in imgIds:
            try:
                res[imgId] = self.results[imgId]
                gts[imgId] = self.gts[imgId]
            except:
                continue

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        
        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            # (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
            ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.2f"%(m, sc*100))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.2f"%(method, score*100))
        
        print('Finished')
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(sorted(imgIds), scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            if method == 'SPICE':
                self.imgToEval[imgId][method] = score['All']['f']
            else:
                self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [self.imgToEval[imgId] for \
            imgId in sorted(self.imgToEval.keys())]


if __name__ == '__main__':
    #TODO set your data and results path
    whoops_path = '/path/to/your/whoops/'
    data_root = whoops_path + 'results/'
    anno_file = whoops_path + 'whoops_dataset.csv'
    blip_results_file = whoops_path + 'whoops_dataset_for_evaluation.csv'
    subtypes=['evalImgs', 'eval']
    
    model_name = 'llava15'
    # model_name = 'instructblip'
    
    time_dir = 'yymmdd-hhmmss'
    dataDir = data_root + time_dir 
    
    # decode_method = 'sample'
    decode_method = 'greedy'
    
    # decode_assist = 'w-rancd'
    decode_assist = 'wo-cd'
    
    result_file = dataDir + f'/{model_name}_whoops_zeroshot_captions_image_{decode_method}_{decode_assist}.json'
    [evalImgsFile, evalFile]= \
    ['%s/%s_whoops_zeroshot_captions_%s.json'%(dataDir, model_name, subtype) for subtype in subtypes]
    
    whoops_data = load_data_for_whoops(whoops_path)
    gts = {}
    for line in tqdm(whoops_data):
        image_id = line['image_id']
        if image_id not in gts.keys():
            gts[image_id] = []
        all_gt_caps = line["crowd_captions"]
        assert len(all_gt_caps) == 5
        for gt_cap in all_gt_caps:
            gts[image_id] += [{'caption': gt_cap.rstrip(".") if gt_cap.endswith(".") else gt_cap}]
            
    results = {}
    for res in open(os.path.expanduser(result_file), "r"):
        res_dict = json.loads(res)
        image_id = res_dict["image_id"]
        if image_id not in results.keys():
            results[image_id] = []
        pred_cap = res_dict["caption_pred"]
        results[image_id] = [{'caption': pred_cap.rstrip(".") if pred_cap.endswith(".") else pred_cap}]
        
    # try read the officially released result file
    # whoops_blip result is downloaded from 
    # whoops official website https://whoops-benchmark.github.io/
    # whoops_blip = load_blip_results_for_whoops(whoops_path)
    # blip_caps_results = {}
    # blip_caps_gts = {}
    # for line in tqdm(whoops_blip):  # bs=1 only
    #     image_id = line['image_id']
    #     if image_id not in blip_caps_gts.keys():
    #         blip_caps_gts[image_id] = []
    #     if image_id not in blip_caps_results.keys():
    #         blip_caps_results[image_id] = []
    #     blip_cap = line["blip_captioning"]
    #     blip_caps_results[image_id] = [{'caption': blip_cap.rstrip(".") if blip_cap.endswith(".") else blip_cap}]
    #     all_gt_caps = line["crowd_captions"]
    #     assert len(all_gt_caps) == 5
    #     for gt_cap in all_gt_caps:
    #         blip_caps_gts[image_id] += [{'caption': gt_cap.rstrip(".") if gt_cap.endswith(".") else gt_cap}]
            
    eval_data = ['all']
    whoopsEval = WhoopsEvalCap(results, gts, data=eval_data)
    # whoopsEval = WhoopsEvalCap(blip_caps_results, blip_caps_gts, data=eval_data)

    whoopsEval.evaluate()
    # save metric files
    json.dump(whoopsEval.evalImgs, open(evalImgsFile, 'w'))
    json.dump(whoopsEval.eval,     open(evalFile, 'w'))
    
    # print output evaluation scores
    # for metric, score in dramaEval.eval.items():
    #     print( '%s: %.3f'%(metric, score))
    
    # plot SPICE histogram
    spiceScores = [eva['SPICE'] for eva in whoopsEval.evalImgs]
    plt.figure()
    plt.hist(spiceScores)
    plt.title('Histogram of SPICE Scores', fontsize=20)
    plt.xlabel('SPICE score', fontsize=20)
    plt.ylabel('result counts', fontsize=20)
    plt.savefig(dataDir + f'/spice_hist.png')
    
    # plot CIDEr histogram
    ciderScores = [eva['CIDEr'] for eva in whoopsEval.evalImgs]
    plt.figure()
    plt.hist(ciderScores)
    plt.title('Histogram of CIDEr Scores', fontsize=20)
    plt.xlabel('CIDEr score', fontsize=20)
    plt.ylabel('result counts', fontsize=20)
    plt.savefig(dataDir + f'/cider_hist.png')
    
    