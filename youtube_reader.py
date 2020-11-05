import tensorflow as tf
import decoder_and_encoder
import numpy as np
import evaluation
from youtube_transcript_api import YouTubeTranscriptApi

_VOCAB_FILE = 'ckpt/c4.unigram.newline.10pct.96000.model'

encoder = decoder_and_encoder.generate_text_encoder("sentencepiece", 
                                                    _VOCAB_FILE)
input_size_limit = {
    'cnn_dailymail': 1024
}

if __name__ == '__main__':
    import argparse
        
    parser = argparse.ArgumentParser(description="Process model and article related arguments.")
    # 1. model_dir
    parser.add_argument("--model_dir", help="path of your model", default="model/")
    # 2. model_name
    parser.add_argument("--model_name", help="name of your model", default="cnn_dailymail")
    # 3. youtube_id
    parser.add_argument("--youtube_id", help="input youtube", default="3Rzi11Hvyh0")
    args = parser.parse_args()
    
    transcript_list = YouTubeTranscriptApi.list_transcripts(args.youtube_id)
    
    paragraph = ''
    
    for transcript in transcript_list:
        
        if transcript.language_code == 'en':
            
            for transcript_sentence in transcript.fetch():
                sentence = transcript_sentence['text']
                paragraph += sentence + '\n'
            
            paragraph = paragraph.replace('\n', ' ')
            paragraph = paragraph.replace('. ', '.\n\n')
            sentence_to_capitalize = paragraph.split('.\n\n')
            new_paragraph = []
            for sentence in sentence_to_capitalize:
                new_paragraph.append(sentence.capitalize())
            article = '.\n\n'.join(new_paragraph)
            
            print(article)
            
            break    
        
    input_ids = encoder.encode(article)
        
    idx = len(input_ids)
    if idx > input_size_limit[args.model_name]:
        idx = input_size_limit[args.model_name]
        
    inputs = np.zeros(input_size_limit[args.model_name])
    inputs[:idx] = input_ids[:idx]
        
    imported = tf.saved_model.load(args.model_dir, tags='serve')
    # Initialize a TF example
    example = tf.train.Example()
    # Load inputs into example feature dictionary
    example.features.feature["inputs"].int64_list.value.extend(inputs.astype(int))
    # Generate outputs by signature serving_default
    output = imported.signatures['serving_default'](examples=tf.constant([example.SerializeToString()]))
        
    summary = evaluation.ids2str(encoder, output['outputs'].numpy(), None)
        
    print(summary)

