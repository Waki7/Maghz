# Installation:

Create a python environment, I like using conda. 

pip install -e C:\Users\ceyer\OneDrive\Documents\Projects\Maghz

If you have a gpu:
    
    pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0 13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    
    pip3 install torchdata==0.5.1 torchtext==0.14.1

download datasets: 
enron: https://bailando.berkeley.edu/enron/enron_with_categories.tar.gz
aus legal case reports: https://archive.ics.uci.edu/static/public/239/legal+case+reports.zip




running tensorboard 
tensorboard --logdir=index_dir/main --bind_all










    # BART
'facebook/bart-large'
'facebook/bart-large-xsum'  # has a bug where it doesn't have 'mask' in its embedding table, or something like that
'facebook/bart-base'
'allenai/bart-large-multi_lexsum-long-short'

    # LED
'allenai/led-base-16384-multi_lexsum-source-long'


    # Mistral
'mistralai/Mistral-7B-v0.1'
"openchat/openchat_3.5"
