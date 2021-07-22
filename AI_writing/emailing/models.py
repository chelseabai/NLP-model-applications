from django.db import models
from django.template.defaultfilters import slugify
from django.contrib.auth.models import User
from django.urls import reverse
import synonyms
import os
import argparse
import logging
import torch
import torch.nn.functional as F
from .modeling_gpt2 import GPT2LMHeadModel, GPT2Config
from .tokenization_bert import BertTokenizer
import math
from argparse import Namespace
from django.shortcuts import render
from django import forms
import json
import heapq
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, HttpResponseRedirect

# Create your models here.

# AI model output control functions.

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def top_kp_search(args, model, tokenizer, curr_input_tensor):
    sep_token_id = tokenizer.vocab["[SEP]"]
    unk_token_id = tokenizer.vocab["[UNK]"]

    generated = []
    for _ in range(args.max_pred_len):
        with torch.no_grad():
            outputs = model(input_ids=curr_input_tensor)
            next_token_logits = outputs[0][-1, :]
            for id in set(generated):
                next_token_logits[id] /= args.repetition_penalty
            next_token_logits = next_token_logits / args.temperature
            next_token_logits[unk_token_id] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == sep_token_id:
                break
            generated.append(next_token.item())
            curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
    return "".join([token.replace("##","") for token in tokenizer.convert_ids_to_tokens(generated)])


def k_best_outputs(curr_input_tensor, outputs, log_scores, index, beam_size):
    probs, ix = outputs[:, index-1].data.topk(beam_size)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(beam_size, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(beam_size)

    row = k_ix // beam_size
    col = k_ix % beam_size

    curr_input_tensor[:, :index] = curr_input_tensor[row, :index]
    curr_input_tensor[:, index] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return curr_input_tensor, log_scores

#清洗标点符号，适用于广告词，正向去找标点符号，返回句子较为简洁
def filtering_outputs(inputstring):
  output=[]
  inputstring=inputstring.replace("，，","，")
  inputstring=inputstring.replace("......","。")
  inputstring=inputstring.replace("，。","。")
  inputstring=inputstring.replace("--"," ")
  inputstring=inputstring.replace("(","")
  inputstring=inputstring.replace(")","")
  inputstring=inputstring.replace("[MASK]", "") 

  #print(inputstring.find("。"))
  if inputstring.find("。")!= -1 or inputstring.find("！")!= -1 or inputstring.find("？")!= -1:
    indexofj = inputstring.find("。")
    #print(indexofj)
    indexofg = inputstring.find("！")
    #print(indexofg)
    indexofw  = inputstring.find("？")
    #print(indexofw)
    for parameter in range(0, max([indexofj,indexofg,indexofw])+1):
      output.append(inputstring[parameter])

  else:
      output=inputstring

  return "".join(output)



#清洗标点符号函数，适用于邮件，反向找标点符号，最大程度的保留原模型的输入
def fliteroutput(inputstring):
  output=[]
  inputstring=inputstring.replace("，，","，")
  inputstring=inputstring.replace("...","。")
  inputstring=inputstring.replace("，。","。")
  inputstring=inputstring.replace("--"," ")
  ##inputstring=inputstring.replace("（","")
  ##inputstring=inputstring.replace("）","")
  inputstring=inputstring.replace("[MASK]", "")
  inputstring=inputstring.replace("[CLS]", "")  

  #print(inputstring.find("。"))
  if inputstring.rfind("。")!= -1 or inputstring.rfind("！")!= -1 or inputstring.rfind("？")!= -1:
    indexofj = inputstring.rfind("。")
    #print(indexofj)
    indexofg = inputstring.rfind("！")
    #print(indexofg)
    indexofw  = inputstring.rfind("？")
    #print(indexofw)
    for parameter in range(0, max([indexofj,indexofg,indexofw])+1):
      output.append(inputstring[parameter])

  elif inputstring.rfind("，")!= -1:
    for parameter in range(0, inputstring.rfind("，")):
      output.append(inputstring[parameter])
    output.append("。")
  else:
    output=inputstring+"..."

  return "".join(output)

def convert_list_to_string(org_list, seperator=' '):
    """ Convert list to string, by joining all item in list with given separator.
        Returns the concatenated string """
    return seperator.join(org_list)

#返回相关度最高的一个句子，适用于邮件生成
def outputtopsentence(topictext,textgeneratebyGPT):
  smallsentence=[]
  sentencescore=[]
  if "。"in textgeneratebyGPT:
    print("ja")
    x = textgeneratebyGPT.split("。")
    print(x)
    while("" in x): 
      x.remove("")
    
    #计算句子内部相关度
    for string in x:

      if "[MASK][CLS]" in string:
        print("yes")
        x.remove(string)
      else:
        if string:
          print("finished split")
          smallsentence.append(string.split("，"))
        else:
          print("Not finished split, It is a whole sentence")
          return textgeneratebyGPT
    for i in range(0, len(smallsentence)):
      print(smallsentence[i])
      score=0
      for j in range(0,len(smallsentence[i])-1):
        #print(smallsentence[i][j])
        #print(smallsentence[i][j+1])
        score=score+synonyms.compare(smallsentence[i][j], smallsentence[i][j+1], seg=True)+synonyms.compare(smallsentence[i][j], topictext, seg=True)
        print(score)
      sentencescore.append(score)
    print(sentencescore)
    #twolargestscoreindex=heapq.nlargest(2, range(len(sentencescore)), key=sentencescore.__getitem_
    filterstring=convert_list_to_string(smallsentence[sentencescore.index(max(sentencescore))],',')+"。"
    #filterstring=smallsentence[twolargestscoreindex[0]]+smallsentence[twolargestscoreindex[1]]

    print("The concatenated string is : ", filterstring) 
    #return smallsentence[sentencescore.index(max(sentencescore))]
    return filterstring
  else:
    if "[MASK][CLS]" in textgeneratebyGPT:
      print("zes") 
      textgeneratebyGPT = textgeneratebyGPT.replace("[MASK][CLS]", "") 
      return textgeneratebyGPT
    else:
      textgeneratebyGPT ="您看这样可以吗？"
      return textgeneratebyGPT


def beam_search(args, model, tokenizer, input_tensor):
    sep_token_id = tokenizer.vocab["[SEP]"]
    unk_token_id = tokenizer.vocab["[UNK]"]
    beam_size = args.beam_size
    log_scores = torch.FloatTensor([0.0]*beam_size).unsqueeze(0)
    input_tensor_len = input_tensor.size(-1)

    ind = None
    curr_input_tensor = torch.zeros(beam_size, input_tensor_len + args.max_pred_len).long().to(args.device)
    curr_input_tensor[:, :input_tensor_len] = input_tensor
    for index in range(args.max_pred_len):
        with torch.no_grad():
            outputs = model(input_ids=curr_input_tensor)
            outputs = F.softmax(outputs[0], dim=-1)
            curr_input_tensor, log_scores = k_best_outputs(curr_input_tensor, outputs, log_scores, input_tensor_len+index, beam_size)

            ones = (curr_input_tensor == sep_token_id).nonzero()  # Occurrences of end symbols for all input sentences.
            sentence_lengths = torch.zeros(len(curr_input_tensor), dtype=torch.long)
            for vec in ones:
                sentence_lengths[vec[0]] += 1

            num_finished_sentences = len([s for s in sentence_lengths if s == 2])

            if num_finished_sentences == beam_size:
                alpha = 0.7
                div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break

    if ind is None:
        ind = 0
    best_output = curr_input_tensor[ind]
    sep_indexs = [x[0].item() for x in (best_output == sep_token_id).nonzero()]
    if len(sep_indexs) == 1:
        sep_indexs.append(input_tensor_len+args.max_pred_len)
    generated = best_output[input_tensor_len: sep_indexs[1]].detach().cpu().numpy()
    return "".join([token.replace("##","") for token in tokenizer.convert_ids_to_tokens(generated)])


def labelofcontent(topictext):
  topiclist = ['会议','报告','请假','面试','通知']
  score=[]
  if topictext:
    for topic in topiclist:
      score.append(synonyms.compare(topic, topictext, seg=True))
    print(score)
    print("返回的主题是：", topiclist[score.index(max(score))])
    return score.index(max(score))
  else:
    print("请输入主题词")
