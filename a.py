import os
import logging
logging.getLogger('tensorflow').disabled = True

from pytube import YouTube
from pydub import AudioSegment
import assemblyai as aai
from transformers import pipeline
import random
import string
from nltk.corpus import stopwords
import pke
from generate_summary import Summary
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor 
from extract_keywords import final_keywords
import requests
import re
import random
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from nltk.corpus import wordnet 
from find_sentances import extract_sentences
import nltk
import pandas as pd
from summarizer import TransformerSummarizer

# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# Function to convert video to audio
def video_to_audio(yt_url):
    yt = YouTube(yt_url)
    ys = yt.streams.filter(only_audio=True).first()
    ad = ys.download()
    base, ext = os.path.splitext(ad)
    audio = AudioSegment.from_file(ad)
    audio.export(base+'.mp3', format='mp3')
    os.remove(ad)
    print("Download Complete!")
    return base+'.mp3'

# Function to transcribe audio to text
def audio_to_text(filepath):
    aai.settings.api_key = "aded0399c8dc45e5b605a8856af99fd6"
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(filepath)

    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
        return
    else:
        return transcript.text



def Summary(text):
    model=TransformerSummarizer(transformer_type="XLNet", transformer_model_key="xlnet-base-cased")
    result = model(text, min_length=60, max_length=500, ratio=0.4)
    summary = "".join(result)
    return summary


def extracting_keywords(text):
    #list to store our keywords
    print("1.Extracting Keywords(ProperNoun) from Fulltext...")
    keywords = []
    #initialize extractor
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(text)
    #we want to extract proper noun
    pos = {'PROPN'}
    
    #define stopwords and others
    stoplist = list(string.punctuation)
    stoplist+= stopwords.words('english')
    stoplist+= ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    
    extractor.candidate_selection(pos=pos)
    
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=15)
    for i in keyphrases:
        keywords.append(i[0])
    return keywords

def final_keywords(text,quantity):
    
    keywords_from_fulltext = extracting_keywords(text)
    if(quantity=='0'):
        print("2(a).Generating summary with Transformers.Pls wait!!")
        generated_summary = Summary(text)
        filtered_keywords = []
        for i in keywords_from_fulltext:
            if i.lower() in generated_summary.lower():
                filtered_keywords.append(i)
        print("2(b).Selected Keywords from summary :",filtered_keywords)
        return filtered_keywords,generated_summary
    else:
        print("2.Selected Keywords from Full Text :",keywords_from_fulltext)
        return keywords_from_fulltext,text
        

def set_sentances(text):
    print("3.Selecting Sentences based on keywords...")
    sentences = [sent_tokenize(text)]
    # nested list to single list
    sentences = [i for sent in sentences for i in sent]
    
    #remove short sentences
    sentences = [sent.strip() for sent in sentences if len(sent)>20]
    #print(sentences)
    return sentences




def extract_sentences(text,quantity):
    keywords,text = final_keywords(text,quantity)
    key_processor = KeywordProcessor()
    filtered_sentences = {}
    
    #adding keywords to processor and to dict
    for i in keywords:
        filtered_sentences[i]=[]
        key_processor.add_keyword(i)
        
    #calling fn to set sentences from summary text
    sentences = set_sentances(text)
    print("4.Filtering sentences...")
    #extracting sentences with given keywords and add to dict keys
    for sent in sentences:
        keyword_searched = key_processor.extract_keywords(sent)
        for key in keyword_searched:
            filtered_sentences[key].append(sent)
    filtered_sentences = dict([(key,val) for key,val in filtered_sentences.items() if(val)])
    
    #sorting with longest sentence of given keyword on top
    for i in filtered_sentences.keys():
        values = filtered_sentences[i]            
        values = sorted(values,key=len,reverse=True)
        filtered_sentences[i] = values
        
    print(filtered_sentences)
    return filtered_sentences

def wordnet_distractors(syon, word):
    print("6.Obtaining relative options from Wordnet...")
    distractors = []
    word = word.lower()
    original_word = word
    #checking if word is more than one word then make it one word with _
    if len(word.split())>0:
        word = word.replace(" ","_")      
    hypersyon = syon.hypernyms()
    if(len(hypersyon)==0):
        return distractors
    for i in hypersyon[0].hyponyms():
        name = i.lemmas()[0].name()       
        if(name==original_word):
            continue
        name = name.replace("_"," ")
        name = " ".join(i.capitalize() for i in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors


def conceptnet_distractors(word):
    print("6.Obtaining relative options from ConceptNet...")
    word = word.lower()
    original_word = word
    if (len(word.split())>0):
        word = word.replace(" ","_")
    distractor_list = [] 
    url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
    obj = requests.get(url).json()
    for edge in obj['edges']:
        link = edge['end']['term'] 
        url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)                 
    return distractor_list

def word_sense(sentence, keyword):
    print("5.Getting word sense to obtain best MCQ options with WordNet...")
    word = keyword.lower()
    if len(word.split())>0:
        word = word.replace(" ","_")  
    syon_sets = wordnet.synsets(word,'n')
    if syon_sets:
        try:
            wup = max_similarity(sentence, word, 'wup', pos='n')
            adapted_lesk_output =  adapted_lesk(sentence, word, pos='n')
            lowest_index = min(syon_sets.index(wup),syon_sets.index(adapted_lesk_output))
            return syon_sets[lowest_index]
        except:
            return syon_sets[0]           
    else:
        return None

    
def display(text, quantity):   
    filtered_sentences = extract_sentences(text, quantity)    
    options_for_mcq = {}
    for keyword in filtered_sentences:
        wordsense = word_sense(filtered_sentences[keyword][0],keyword)
        if wordsense:
           distractors = wordnet_distractors(wordsense,keyword) 
           if len(distractors)>0:
                options_for_mcq[keyword]=distractors
           if len(distractors)<4:
               distractors = conceptnet_distractors(keyword)
               if len(distractors)>0:
                    options_for_mcq[keyword]=distractors                   
        else:
            distractors = conceptnet_distractors(keyword)
            if len(distractors)>0:
                options_for_mcq[keyword] = distractors
    print("7. Creating JSON response for API...")
    df = pd.DataFrame()
    cols = ['question','options','extras','answer']    
    index = 1
    print ("****************************")
    print ("NOTE: Human intervention is required to correct some of the generated MCQ's ")
    print ("****************************\n\n")
    for i in options_for_mcq:
        sentence = filtered_sentences[i][0]
        sentence = sentence.replace("\n",'')
        pattern = re.compile(i, re.IGNORECASE)
        output = pattern.sub( " __ ", sentence)
        print ("%s)"%(index),output)
        options = [i.capitalize()] + options_for_mcq[i]
        top4 = options[:4]
        random.shuffle(top4)
        optionsno = ['a','b','c','d']
        for idx,choice in enumerate(top4):
            print ("\t",optionsno[idx],")"," ",choice)
        print ("\nMore options: ", options[4:8],"\n\n")
        df = df._append(pd.DataFrame([[output,top4,options[4:8],i.capitalize()]],columns=cols))
        index = index + 1               
    df.to_json('response.json',orient='records',force_ascii=False)


if __name__ == "__main__":
    yt_url = input("Enter YouTube URL: ")
    quantity = input("Enter 0 for summary-based keywords or 1 for full-text keywords: ")
    
    audio_file = video_to_audio(yt_url)
    text = audio_to_text(audio_file)
    print(text)
    
    if text:
        display(text, quantity)
    else:
        print("shit")    
        