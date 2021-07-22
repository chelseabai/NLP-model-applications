import os
import argparse
import logging
import torch
import torch.nn.functional as F
from .modeling_gpt2 import  GPT2Config, GPT2LMHeadModel
#from transformers import GPT2LMHeadModel
from .tokenization_bert import BertTokenizer
import math
from argparse import Namespace
from django.shortcuts import render
from django import forms
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, HttpResponseRedirect
#from urls import config, tokenizer, model
from .models import top_k_top_p_filtering, top_kp_search, k_best_outputs, filtering_outputs, beam_search, labelofcontent, fliteroutput, outputtopsentence
#### model functions ####


logger = logging.getLogger(__name__)

#Loading GPT2 model
args = Namespace(max_pred_len=100, model_path='writing/GPT2-large-v1.0', repetition_penalty=2, search='top_kp', temperature=0.9, topk=10, topp=0)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = GPT2Config.from_pretrained(args.model_path)
tokenizer = BertTokenizer.from_pretrained(os.path.join(args.model_path, "vocab.txt"))
model_GPT2 = GPT2LMHeadModel.from_pretrained(args.model_path, config=config)

#Loading 清源GPT2 model

# #Loading finetuned model for email
args_finetune = Namespace(max_pred_len=100, model_path='writing/GPT2-large-v1.0', repetition_penalty=10, search='top_kp', temperature=0.9, topk=10, topp=0)
args_finetune.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_finetune = GPT2Config.from_pretrained(args_finetune.model_path)
tokenizer_finetune = BertTokenizer.from_pretrained(os.path.join(args_finetune.model_path, "vocab.txt"))
model_finetune = GPT2LMHeadModel.from_pretrained(args_finetune.model_path, config=config_finetune)

#Loading finetuned model for advertisement
args_finetune_ads = Namespace(max_pred_len=30, model_path='writing/advertise-model', repetition_penalty=2, search='top_kp', temperature=0.9, topk=10, topp=0)
args_finetune_ads.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_finetune_ads = GPT2Config.from_pretrained(args_finetune_ads.model_path)
tokenizer_finetune_ads = BertTokenizer.from_pretrained(os.path.join(args_finetune_ads.model_path, "vocab.txt"))
model_finetune_ads = GPT2LMHeadModel.from_pretrained(args_finetune_ads.model_path, config=config_finetune_ads)


def generate_email(content_text):

    global args_finetune
    global config_finetune
    global model_finetune
    global tokenizer_finetune


    model_finetune.to(args_finetune.device)

    #Prompt给语料为appellation，topic_text，content_text 利用[","]分割
    token_ids = tokenizer_finetune.convert_tokens_to_ids(["[CLS]"])

    token_ids.extend(tokenizer_finetune.convert_tokens_to_ids(tokenizer_finetune.tokenize(content_text)))

    token_ids.extend(tokenizer_finetune.convert_tokens_to_ids(["[SEP]"]))

    #print("分词ID: {}".format(token_ids))

    input_features = torch.tensor(token_ids, dtype=torch.long).to(args_finetune.device)
    if args.search == "top_kp":
      gen_text = top_kp_search(args_finetune, model_finetune, tokenizer_finetune, input_features)
      print("GPT原始大模型生成的句子是:", gen_text)
      return gen_text
    else:
      raise Exception
      gen_text = "Error!"
      return gen_text

def generate_ads(input_text):

    global args_finetune_ads
    global config_finetune_ads
    global model_finetune_ads
    global tokenizer_finetune_ads

    model_finetune_ads.to(args_finetune_ads.device)

    token_ids = tokenizer_finetune_ads.convert_tokens_to_ids(["[CLS]"])

    token_ids.extend(tokenizer_finetune_ads.convert_tokens_to_ids(tokenizer_finetune_ads.tokenize(input_text)))

    token_ids.extend(tokenizer_finetune_ads.convert_tokens_to_ids(["[SEP]"]))

    input_features = torch.tensor(token_ids, dtype=torch.long).to(args_finetune_ads.device)
    if args_finetune_ads.search == "top_kp":
      gen_text = filtering_outputs(top_kp_search(args_finetune_ads, model_finetune_ads, tokenizer_finetune_ads, input_features))

      return gen_text
    else:
      raise Exception
      gen_text = "Error!"
      return gen_text

def customise(model_type,content_text,tmp,topk,topp,length):
    global config
    global model_GPT2
    global tokenizer
    if model_type == 1:
        args = Namespace(max_pred_len=length, model_path='writing/GPT2-large-v1.0', repetition_penalty=1, search='top_kp', temperature=tmp, topk=topk, topp=topp)
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_GPT2.to(args.device)
        token_ids = tokenizer.convert_tokens_to_ids(["[CLS]"])
        token_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(content_text)))
        token_ids.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))
        input_features = torch.tensor(token_ids, dtype=torch.long).to(args.device)
        if args.search == "top_kp":
          gen_text = fliteroutput(top_kp_search(args, model_GPT2, tokenizer, input_features))
          return gen_text
        else:
          raise Exception
          gen_text = "Error!"
          return gen_text
    else:
        gen_text = "清源还没接上！"
        return gen_text

def excel(request):
    return render(request, "emailing/excel.html")

def aboutus(request):
    return render(request, "emailing/aboutus.html")

@csrf_exempt
def email(request):
    if request.method=='POST':
        if request.POST.get('classification') == 'true':
            email_subject  = request.POST.get('subject')
            index = labelofcontent(email_subject)
            data = {"index":index}
            return HttpResponse(json.dumps(data,ensure_ascii=False))
        else:
            content_text = request.POST.get('content_text')
            content_text = eval(content_text)
            email_subject  = request.POST.get('subject')
            responses = []
            for i in content_text:
                i = email_subject + "," + i
                ret = outputtopsentence(email_subject, fliteroutput(generate_email(i))) #现在在外围只嵌套一层函数过滤输出，如需控制只输出一句
                responses.append(ret)       #使用方法outputtopsentence（主题词诸如会议，GPT生成的text） outputtopsentence(topic, fliteroutput(main(i)))
                data = {"responses":responses}
            return HttpResponse(json.dumps(data,ensure_ascii=False))
    return render(request, "emailing/email.html")

@csrf_exempt
def gptmodel(request):
    if request.method=='POST':
        model_type = int(request.POST.get('model_type'))
        content_text = request.POST.get('content_text')
        tmp = float(request.POST.get('tmp'))
        topp = float(request.POST.get('topp'))
        topk = int(request.POST.get('topk'))
        length = int(request.POST.get('length'))
        response = customise(model_type,content_text,tmp,topk,topp,length)
        data = {"response":response}
        return HttpResponse(json.dumps(data,ensure_ascii=False))
    return render(request, "emailing/model.html")

@csrf_exempt
def advertise(request):
    if request.method=='POST':
        content = request.POST.get('content')
        input_string = content
        response1 = generate_ads(input_string)
        response2 = generate_ads(input_string)
        response3 = generate_ads(input_string)
        data = {"response1":response1, "response2":response2,"response3":response3}
        return HttpResponse(json.dumps(data,ensure_ascii=False))
    return render(request, "emailing/advertise.html")
